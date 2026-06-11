#!/usr/bin/env python3
import struct
import shutil
import os
import subprocess
import sqlite3
import glob
import argparse
import sys
from scipy.signal import butter, filtfilt
import numpy as np

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

def load_audio_ffmpeg(file_path: str, end_time: float, target_sr: int = 22050) -> np.ndarray:
    # Read mono raw float32 samples directly from sample zero to ensure 100% sample accuracy (zero seek error)
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", file_path,
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(target_sr),
        "-"
    ]
    # Guarantee max_bytes is a multiple of 4 (each float32 sample is 4 bytes)
    max_bytes = int(end_time * target_sr) * 4
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    raw_audio, _ = process.communicate()
    raw_audio = raw_audio[:max_bytes]
    return np.frombuffer(raw_audio, dtype=np.float32)

def apply_kick_filter(signal: np.ndarray, sr: int) -> np.ndarray:
    low = 40.0
    high = 150.0
    nyq = 0.5 * sr
    low_cut = low / nyq
    high_cut = high / nyq
    b, a = butter(4, [low_cut, high_cut], btype='band')
    return filtfilt(b, a, signal)

def snap_to_daw_bpm(bpm: float) -> float:
    nearest_int = round(bpm)
    if abs(bpm - nearest_int) <= 0.08:
        return float(nearest_int)
    nearest_half = round(bpm * 2) / 2.0
    if abs(bpm - nearest_half) <= 0.08:
        return float(nearest_half)
    return bpm

def detect_bpm_and_phase(signal: np.ndarray, sr: int) -> tuple[float, int]:
    # 1. Compute high-resolution envelope
    env = np.abs(signal)
    # Smooth with 10ms window to remove sub-sample high frequency ripples
    window_len = int(0.010 * sr) | 1
    window = np.ones(window_len) / window_len
    env_smoothed = np.convolve(env, window, mode='same')
    
    # First-order derivative + Half-wave rectification (gives crisp kick onset spikes)
    onset_env = np.diff(env_smoothed)
    onset_env = np.append(onset_env, 0.0) # Pad diff to match signal length
    onset_env = np.maximum(0.0, onset_env)
    
    # 2. Decimate envelope for fast FFT search (target ~2205 Hz)
    decim_factor = int(sr / 2205) if sr > 2205 else 1
    env_decim = onset_env[::decim_factor]
    decim_sr = sr / decim_factor
    
    # Remove DC offset (mean subtraction)
    env_decim_centered = env_decim - np.mean(env_decim)
    n = len(env_decim_centered)
    
    # Compute FFT of decimated envelope
    f_env = np.fft.fft(env_decim_centered)
    
    # Pre-apply smoothing window in Fourier space (Convolution Theorem)
    w_len = int(0.015 * decim_sr) | 1
    w = np.hamming(w_len)
    w_padded = np.zeros(n)
    w_padded[:len(w)] = w
    w_padded = np.roll(w_padded, -len(w)//2) # Center the window
    f_win = np.fft.fft(w_padded)
    f_env_smoothed = f_env * np.conj(f_win)
    
    # Pass 1: Coarse search (130 to 165 BPM, step 0.1)
    best_coarse_score = -1.0
    best_coarse_bpm = 140.0
    
    coarse_bpms = np.arange(130.0, 165.0, 0.1)
    for bpm in coarse_bpms:
        beat_samples = (60.0 / bpm) * decim_sr
        # Vectorized pulse train generation
        beat_indices = (np.arange(int(n / beat_samples)) * beat_samples).astype(int)
        beat_indices = beat_indices[beat_indices < n]
        template = np.zeros(n)
        template[beat_indices] = 1.0
        
        f_temp = np.fft.fft(template)
        corr = np.fft.ifft(f_env_smoothed * np.conj(f_temp)).real
        score = np.max(corr)
        if score > best_coarse_score:
            best_coarse_score = score
            best_coarse_bpm = bpm
            
    # Pass 2: Ultra-fine search (+- 0.15 around best coarse, step 0.01)
    best_fine_score = -1.0
    best_fine_bpm = best_coarse_bpm
    best_offset_decim = 0
    
    fine_bpms = np.arange(best_coarse_bpm - 0.15, best_coarse_bpm + 0.15, 0.01)
    for bpm in fine_bpms:
        beat_samples = (60.0 / bpm) * decim_sr
        beat_indices = (np.arange(int(n / beat_samples)) * beat_samples).astype(int)
        beat_indices = beat_indices[beat_indices < n]
        template = np.zeros(n)
        template[beat_indices] = 1.0
        
        f_temp = np.fft.fft(template)
        corr = np.fft.ifft(f_env_smoothed * np.conj(f_temp)).real
        score = np.max(corr)
        if score > best_fine_score:
            best_fine_score = score
            best_fine_bpm = bpm
            best_offset_decim = np.argmax(corr)
            
    # Apply DAW rounding heuristics to candidate
    final_bpm = snap_to_daw_bpm(best_fine_bpm)
    
    # If the snapped BPM is different from the best fine BPM, re-run correlation to find perfect phase
    if abs(final_bpm - best_fine_bpm) > 1e-5:
        beat_samples = (60.0 / final_bpm) * decim_sr
        beat_indices = (np.arange(int(n / beat_samples)) * beat_samples).astype(int)
        beat_indices = beat_indices[beat_indices < n]
        template = np.zeros(n)
        template[beat_indices] = 1.0
        f_temp = np.fft.fft(template)
        corr = np.fft.ifft(f_env_smoothed * np.conj(f_temp)).real
        best_offset_decim = np.argmax(corr)
        
    # 4. Map the decimated offset back to high-resolution samples (in original signal)
    best_offset_high_res = best_offset_decim * decim_factor
    
    # Local high-res refinement on original high-res envelope
    local_window = int( (60.0 / final_bpm) * sr * 0.05 ) # 5% of a beat
    start_search = max(0, best_offset_high_res - local_window)
    end_search = min(len(env), best_offset_high_res + local_window)
    if start_search < end_search:
        best_offset_high_res = start_search + np.argmax(onset_env[start_search:end_search])
        
    return final_bpm, int(best_offset_high_res)

def detect_bpm(signal: np.ndarray, sr: int) -> float:
    bpm, _ = detect_bpm_and_phase(signal, sr)
    return bpm


def detect_phase(envelope: np.ndarray, bpm: float, sr: int) -> int:
    # 1 beat duration in samples
    beat_samples = (60.0 / bpm) * sr
    
    # Generate synthetic templates (comb filters) to slide over envelope
    # We decimate for faster correlation computation
    decim_factor = int(sr / 500) if sr > 500 else 1
    env_decim = envelope[::decim_factor]
    decim_sr = sr / decim_factor
    decim_beat_samples = (60.0 / bpm) * decim_sr
    
    # We will build a template of impulses
    num_beats = int(len(env_decim) / decim_beat_samples)
    template = np.zeros_like(env_decim)
    for i in range(num_beats):
        beat_idx = int(i * decim_beat_samples)
        if beat_idx < len(template):
            template[beat_idx] = 1.0
            
    # Smooth the template slightly to allow wider matching margin
    kernel_size = int(decim_beat_samples * 0.1) | 1
    kernel = np.hamming(kernel_size)
    template_smooth = np.convolve(template, kernel, mode='same')
    
    # Compute cross-correlation for offsets up to 1 beat
    search_limit = int(decim_beat_samples)
    correlations = []
    
    for offset in range(search_limit):
        # Shift template
        shifted_template = np.roll(template_smooth, offset)
        corr = np.dot(env_decim, shifted_template)
        correlations.append(corr)
        
    best_offset_decim = np.argmax(correlations)
    
    # Reconvert back to original high-res samples
    best_offset_samples = best_offset_decim * decim_factor
    
    # Add a tiny adjustment based on actual envelope local maxima around the detected frame
    local_window = int(beat_samples * 0.05)
    start_search = max(0, best_offset_samples - local_window)
    end_search = min(len(envelope), best_offset_samples + local_window)
    if start_search < end_search:
        best_offset_samples = start_search + np.argmax(envelope[start_search:end_search])
        
    return int(best_offset_samples)


def process_track(file_path: str, db_path: str, dry_run: bool = False, calibrate: float = -0.020):
    print(f"\n[+] Analisando: {os.path.basename(file_path)}")
    
    # 1. Get total duration to find the active middle segment
    cmd_info = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
    try:
        total_duration = float(subprocess.check_output(cmd_info).decode().strip())
    except Exception:
        total_duration = 300.0 # Default fallback
        
    # Analyze 60 seconds starting from 1/3 of the track where beat is established
    start_time = total_duration / 3.0
    end_time = start_time + 60.0
    
    # Load audio
    try:
        sr = 22050
        signal = load_audio_ffmpeg(file_path, end_time=end_time, target_sr=sr)
    except Exception as e:
        print(f"  [Erro] Falha ao decodificar com FFmpeg: {e}")
        return

    if len(signal) == 0:
        print("  [Erro] Sinal de áudio vazio.")
        return
        
    # Slice active segment in Python (sample-accurate!)
    start_sample = int(start_time * sr)
    active_signal = signal[start_sample:]
    
    # Apply filter on the active segment
    filtered = apply_kick_filter(active_signal, sr)
    
    # Detect BPM and Phase
    bpm, best_offset_samples = detect_bpm_and_phase(filtered, sr)
    print(f"  BPM Detectado: {bpm:.2f}")
    
    # Absolute offset in seconds from the beginning of the file, compensated for MP3 decoder delay
    absolute_offset_seconds = start_time + (best_offset_samples / sr) + calibrate
    
    # Modulo beat duration to anchor the grid neatly within the first beat interval near the start
    beat_duration_seconds = 60.0 / bpm
    
    # Safely handle negative offsets if calibration pushes it below 0
    if absolute_offset_seconds < 0:
        absolute_offset_seconds = (absolute_offset_seconds % beat_duration_seconds) + beat_duration_seconds
        
    first_beat_offset_seconds = absolute_offset_seconds % beat_duration_seconds
    
    # First beat frame position (Mixxx tracks this at its own native samplerate)
    # We find native samplerate from DB, or default to 44100
    native_sr = 44100
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT samplerate FROM library JOIN track_locations ON library.location = track_locations.id WHERE track_locations.location = ?;", (file_path,))
        row = cur.fetchone()
        if row and row[0]:
            native_sr = row[0]
        conn.close()
    except Exception:
        pass
        
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
            
            track_id, duration_val, db_sr = row
            if not duration_val:
                duration_val = total_duration
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
    parser = argparse.ArgumentParser(description="Analisador de BPM e Grid de Psytrance de alta precisão para o Mixxx.")
    parser.add_argument("--dir", default="/home/eduardo/Projects/EDU/bandcamp", help="Diretório onde buscar arquivos .mp3")
    parser.add_argument("--db", default="/home/eduardo/.mixxx/mixxxdb.sqlite", help="Caminho do banco de dados do Mixxx")
    parser.add_argument("--dry-run", action="store_true", help="Faz apenas a simulação de análise sem alterar o banco de dados")
    parser.add_argument("--calibrate", type=float, default=-0.020, help="Ajuste fino de calibração em segundos (ex: -0.020 para adiantar o grid em 20ms)")
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
        
    print(f"Encontrados {len(files)} arquivos .mp3. Iniciando processamento...")
    
    for f in files:
        process_track(f, args.db, dry_run=args.dry_run, calibrate=args.calibrate)


if __name__ == "__main__":
    main()

