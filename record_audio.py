import sounddevice as sd
import numpy as np
import webrtcvad
from scipy.io.wavfile import write

def record_audio(fs=16000, frame_duration_ms=30, silence_timeout_ms=5000, vad_aggressiveness=0):
    vad = webrtcvad.Vad(vad_aggressiveness)
    frame_size = int(fs * frame_duration_ms / 1000)  # Frame size in samples
    silence_timeout_frames = silence_timeout_ms // frame_duration_ms

    recorded_frames = []
    silent_frames_count = 0

    def callback(indata, frame_count, time_info, status):
        nonlocal silent_frames_count, recorded_frames
        if status:
            print(f"Error: {status}")

        # Check for speech in the current frame
        if vad.is_speech(indata.tobytes(), fs):
            silent_frames_count = 0
            recorded_frames.append(indata.copy())
        else:
            if silent_frames_count == 0:  # Only append the first silent frame after speech
                recorded_frames.append(indata.copy())
            silent_frames_count += 1

        # Stop recording if the silence has lasted long enough
        if silent_frames_count > silence_timeout_frames:
            raise sd.CallbackStop

    try:
        with sd.InputStream(callback=callback, samplerate=fs, channels=1, dtype='int16', blocksize=frame_size):
            print("Recording... Speak now.")
            sd.sleep(silence_timeout_ms + 1000)  # Extra time to ensure callback has time to stop the stream
    except sd.CallbackStop:
        print("Silence detected, stopping recording.")

    if recorded_frames:
        return np.concatenate(recorded_frames, axis=0)
    else:
        return np.array([], dtype=np.int16)

# if __name__ == "__main__":
#     fs = 16000  # Sample rate
#     audio_data = record_until_silence(fs=fs)
#     if audio_data.size > 0:
#         print("Audio recorded successfully.")
#         # Save the recorded audio to a WAV file
#         filename = "output_vad_detection.wav"
#         write(filename, fs, audio_data.astype(np.int16))
#         print(f"Audio saved to '{filename}'")
#     else:
#         print("No audio recorded.")
