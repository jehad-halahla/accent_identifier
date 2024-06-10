import os
from pydub import AudioSegment

def split_wav_files(input_folder, output_folder, segment_length_ms=600000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_folder, filename)
            audio = AudioSegment.from_wav(filepath)
            file_length_ms = len(audio)
            
            num_segments = (file_length_ms + segment_length_ms - 1) // segment_length_ms  # Round up division

            for i in range(num_segments):
                start_time = i * segment_length_ms
                end_time = min((i + 1) * segment_length_ms, file_length_ms)
                segment = audio[start_time:end_time]
                segment_filename = f"{os.path.splitext(filename)[0]}_part{i+1}.wav"
                segment_filepath = os.path.join(output_folder, segment_filename)
                segment.export(segment_filepath, format="wav")



split_wav_files('training/Ramallah_Reef', 'training_segmented/Ramallah_Reef')
split_wav_files('training/Nablus', 'training_segmented/Nablus')
split_wav_files('training/Jerusalem', 'training_segmented/Jerusalem')
split_wav_files('training/Hebron', 'training_segmented/Hebron')


split_wav_files('testing/Ramallah_Reef', 'testing_segmented/Ramallah_Reef')
split_wav_files('testing/Nablus', 'testing_segmented/Nablus')
split_wav_files('testing/Jerusalem', 'testing_segmented/Jerusalem')
split_wav_files('testing/Hebron', 'testing_segmented/Hebron')
