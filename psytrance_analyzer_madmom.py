#!/usr/bin/env python3
import collections
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence

import numpy as np
np.float = float
np.int = int

import os
import sys
import argparse
import sqlite3
import shutil
import glob
import struct
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

MADMOM_KEY_TO_MIXXX = {
    'C major': (1, 'C'),      'Db major': (2, 'Db'),    'D major': (3, 'D'),
    'Eb major': (4, 'Eb'),    'E major': (5, 'E'),      'F major': (6, 'F'),
    'F# major': (7, 'F#'),    'G major': (8, 'G'),      'Ab major': (9, 'Ab'),
    'A major': (10, 'A'),     'Bb major': (11, 'Bb'),   'B major': (12, 'B'),
    'C minor': (13, 'Cm'),    'C# minor': (14, 'C#m'),  'D minor': (15, 'Dm'),
    'D# minor': (16, 'Ebm'),  'E minor': (17, 'Em'),    'F minor': (18, 'Fm'),
    'F# minor': (19, 'F#m'),  'G minor': (20, 'Gm'),    'G# minor': (21, 'G#m'),
    'A minor': (22, 'Am'),    'Bb minor': (23, 'Bbm'),  'B minor': (24, 'Bm')
}

def encode_varint(val: int) -> bytes:
    res = bytearray()
    while True:
        towrite = val & 0x7f
        val >>= 7
        if val:
            res.append(towrite | 0x80)
        else:
            res.append(towrite)
            break
    return bytes(res)

def serialize_beatgrid(bpm: float, frame_position: int) -> bytes:
    bpm_double = struct.pack('<d', float(bpm))
    bpm_submsg = b'\x09' + bpm_double
    bpm_field = b'\x0a' + encode_varint(len(bpm_submsg)) + bpm_submsg
    
    beat_varint = encode_varint(int(frame_position))
    beat_submsg = b'\x08' + beat_varint
    beat_field = b'\x12' + encode_varint(len(beat_submsg)) + beat_submsg
    
    return bpm_field + beat_field

def backup_database(db_path: str) -> str:
    backup_path = db_path + ".backup"
    if os.path.exists(db_path):
        shutil.copy2(db_path, backup_path)
    return backup_path

def snap_to_daw_bpm(bpm: float) -> float:
    nearest_int = round(bpm)
    if abs(bpm - nearest_int) <= 0.08:
        return float(nearest_int)
    nearest_half = round(bpm * 2) / 2.0
    if abs(bpm - nearest_half) <= 0.08:
        return float(nearest_half)
    return bpm

def detect_bpm_and_phase_madmom(file_path: str, total_duration: float) -> tuple[float, float]:
    import subprocess
    from madmom.audio.signal import Signal
    
    # 1. Load 90 seconds from 1/3 of the track to bypass intro noise
    sr = 44100
    start_time = total_duration / 3.0
    end_time = start_time + 90.0
    
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", file_path,
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(sr),
        "-"
    ]
    # Guarantee max_bytes is a multiple of 4 (each float32 is 4 bytes)
    max_bytes = int(end_time * sr) * 4
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw_audio, _ = process.communicate()
    raw_audio = raw_audio[:max_bytes]
    signal = np.frombuffer(raw_audio, dtype=np.float32)
    
    # Slice active segment (sample-accurate in memory)
    start_sample = int(start_time * sr)
    active_signal = signal[start_sample:]
    
    # Create Madmom Signal object
    madmom_signal = Signal(active_signal, sample_rate=sr)
    
    # 2. Run Deep Learning RNN Beat Activation
    proc = RNNBeatProcessor()
    act = proc(madmom_signal)
    
    # 3. Run Bayesian dynamic beat tracking
    tracker = DBNBeatTrackingProcessor(fps=100)
    beats = tracker(act)
    
    if len(beats) < 4:
        raise ValueError("Não foi possível detectar batidas suficientes para análise.")
        
    # 4. Fit linear regression to enforce constant grid
    # beat_time = offset + index * beat_duration
    x = np.arange(len(beats))
    slope, intercept = np.polyfit(x, beats, 1)
    
    bpm = 60.0 / slope
    # Apply DAW rounding heuristics
    final_bpm = snap_to_daw_bpm(bpm)
    
    # Recalculate optimal phase offset for this exact snapped BPM
    beat_duration = 60.0 / final_bpm
    local_offset = np.mean(beats - x * beat_duration)
    
    # Absolute offset from the beginning of the file is:
    absolute_offset = start_time + local_offset
    
    # Wrap offset to a single beat interval near 0.0 seconds
    absolute_offset = absolute_offset % beat_duration
    
    return final_bpm, absolute_offset

def process_track(file_path: str, db_path: str, dry_run: bool = False, calibrate: float = 0.0):
    print(f"\n[+] Analisando via Deep Learning (Madmom): {os.path.basename(file_path)}")
    
    # Get native samplerate and duration from Mixxx DB, or default to 44100
    native_sr = 44100
    duration_val = 300.0
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('''
            SELECT library.id, library.duration, library.samplerate 
            FROM library 
            JOIN track_locations ON library.location = track_locations.id 
            WHERE track_locations.location = ?;
        ''', (file_path,))
        row = cur.fetchone()
        if row:
            track_id, db_duration, db_sr = row
            if db_duration:
                duration_val = db_duration
            if db_sr:
                native_sr = db_sr
        conn.close()
    except Exception:
        pass

    # Run analysis
    try:
        bpm, first_beat_offset_seconds = detect_bpm_and_phase_madmom(file_path, duration_val)
        # Apply manual calibration fine-tuning if provided
        first_beat_offset_seconds += calibrate
    except Exception as e:
        print(f"  [Erro] Falha na análise neural: {e}")
        return

    print(f"  BPM Detectado: {bpm:.2f}")
    
    beat_duration_seconds = 60.0 / bpm
    # Safely handle negative offsets
    if first_beat_offset_seconds < 0:
        first_beat_offset_seconds = (first_beat_offset_seconds % beat_duration_seconds) + beat_duration_seconds
    else:
        first_beat_offset_seconds = first_beat_offset_seconds % beat_duration_seconds
        
    first_beat_frame = int(first_beat_offset_seconds * native_sr)
    print(f"  Posição da primeira batida: {first_beat_frame} frames ({first_beat_offset_seconds:.4f} segundos em {native_sr}Hz)")
    
    # Serialize beatgrid BLOB
    blob = serialize_beatgrid(bpm, first_beat_frame)
    
    # database write
    if dry_run:
        print("  [Modo de Teste / Dry-run] Banco de dados NÃO modificado.")
    else:
        try:
            # Backup first
            backup_database(db_path)
            
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            
            # Find track ID and duration
            cur.execute("SELECT library.id, library.duration, library.samplerate FROM library JOIN track_locations ON library.location = track_locations.id WHERE track_locations.location = ?;", (file_path,))
            row = cur.fetchone()
            if not row:
                print("  [Aviso] Faixa não encontrada na biblioteca do Mixxx. Certifique-se de adicionar a pasta do seu projeto na biblioteca do Mixxx antes.")
                conn.close()
                return
            
            track_id, db_duration, db_sr = row
            if db_duration:
                duration_val = db_duration
            if db_sr:
                native_sr = db_sr
            
            # Update database
            cur.execute('''
                UPDATE library SET
                    bpm = ?,
                    beats = ?,
                    beats_version = 'BeatGrid-2.0',
                    beats_sub_version = 'rounding=V4|vamp_plugin_id=qm-tempotracker:0',
                    bpm_lock = 1
                WHERE id = ?;
            ''', (bpm, blob, track_id))
            
            # Delete old cues of type 8 (beatgrid start/cues) to prevent overlapping
            cur.execute("DELETE FROM cues WHERE track_id = ? AND type = 8;", (track_id,))
            
            # Add cue type 8 for beatgrid first beat visualization
            # cue positions are saved in stereo samples (mono frames * 2)
            cue_pos = first_beat_frame * 2
            track_len_stereo = int(duration_val * native_sr * 2)
            cur.execute('''
                INSERT INTO cues (track_id, type, position, length, hotcue, label, color)
                VALUES (?, 8, ?, ?, -1, 'PsyBeatGrid', 16744448);
            ''', (track_id, cue_pos, track_len_stereo))
            
            conn.commit()
            conn.close()
            print("  [Sucesso] Metadados gravados e travados no Mixxx.")
        except Exception as e:
            print(f"  [Erro] Falha ao atualizar o banco do Mixxx: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analisador Neural de BPM e Grid de Psytrance de alta precisão para o Mixxx usando Madmom.")
    parser.add_argument("--dir", default="/home/eduardo/Projects/EDU/bandcamp", help="Diretório onde buscar arquivos .mp3")
    parser.add_argument("--db", default="/home/eduardo/.mixxx/mixxxdb.sqlite", help="Caminho do banco de dados do Mixxx")
    parser.add_argument("--dry-run", action="store_true", help="Faz apenas a simulação de análise sem alterar o banco de dados")
    parser.add_argument("--calibrate", type=float, default=0.0, help="Ajuste fino de calibração em segundos (ex: -0.010 para adiantar o grid em 10ms)")
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        print(f"Erro: Banco de dados do Mixxx não encontrado em {args.db}")
        return
        
    print(f"Buscando músicas em: {args.dir}")
    # Search for mp3s recursively
    search_path = os.path.join(args.dir, "**", "*.mp3")
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        print("Nenhuma música .mp3 encontrada no diretório informado.")
        return
        
    print(f"Encontrados {len(files)} arquivos .mp3. Iniciando processamento neural...")
    
    for f in files:
        process_track(f, args.db, dry_run=args.dry_run, calibrate=args.calibrate)

if __name__ == "__main__":
    main()
