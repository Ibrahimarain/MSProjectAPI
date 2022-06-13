from flask import Flask, render_template
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.io import wavfile
import subprocess
import speech_recognition as sr
import nltk as nltk
import re
import ffmpeg
import cv2
import speech_recognition as sr
import os
# from pydub import AudioSegment
# from pydub.silence import split_on_silence
import os
import sys

print(cv2.__version__)

# nltk.download("stopwords")
# nltk.download('punkt')

app = Flask(__name__)

output_file_name = "audio_only"
output_ext = "wav"
frames_output_dir = "./data/frame"
frames_to_detect_dir = "./frames"


def extract_frames():
    print("Extracting Frames")
    cam = cv2.VideoCapture("sampleVideo.mp4")

    try:
        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')

        if not os.path.exists('frames'):
            os.makedirs('frames')

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 0

    while True:

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = frames_output_dir + str(currentframe) + '.jpg'
            # print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


def convert_video_to_audio_ffmpeg(video_file, output_ext="wav"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    filename, ext = os.path.splitext(video_file)
    print("working")
    subprocess.call(["ffmpeg/bin/ffmpeg", "-y", "-i", video_file, f"{output_file_name}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    print("Done")


@app.route('/')
def open_home():  # put application's code here
    extract_frames()
    vf = "sampleVideo.mp4"
    convert_video_to_audio_ffmpeg(vf)
    return render_template('view.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)


# flask run -h 0.0.0.0

# @app.route('/features', methods=['GET'])
@app.route('/features', methods=['GET'])
def extract_features():
    try:
        features = ["product", "height", "brand", "color", "colour", "type", "material", "model", "price", "features",
                    "speciality"]
        worf_quote = ""

        filename = f"{output_file_name}.{output_ext}"
        # subprocess.call(['ffmpeg', '-i', 'voice.mp3',
        #                  'voice.wav'])

        # worf_quote = get_large_audio_transcription('voice2.wav')
        # print(worf_quote)

        r = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            # listen for the data (load audio to memory)
            audio_data = r.record(source)
            # recognize (convert from speech to text)
            # text = r.recognize_google(audio_data)
            # print(text)
            worf_quote = r.recognize_google(audio_data, language="en")
            print(worf_quote)

        worf_quote = remove_brackets(worf_quote)
        worf_quote = worf_quote.lower()
        print(worf_quote)
        words_in_quote = word_tokenize(worf_quote)
        stop_words = set(stopwords.words("english"))
        filtered_list = []

        for word in words_in_quote:
            if word.casefold() not in stop_words:
                filtered_list.append(word)

        featuresFound = []
        featuresIndexes = []
        features_list = []
        response = []
        Output = filtered_list.copy()
        # print(filtered_list)

        # Using iteration
        for elem in filtered_list:
            for n in features:
                if n in elem:
                    featuresFound.append(elem)

        # print("Features Found :", featuresFound)

        for feature in featuresFound:
            featuresIndexes.append(filtered_list.index(feature))

        for index in range(len(featuresIndexes)):
            if index < len(featuresIndexes) - 1:
                indexValue = featuresIndexes[index]
                indexNextValue = featuresIndexes[index + 1]
                features_list.append(filtered_list[indexValue:indexNextValue])

        print("\n")
        for f in features_list:
            print(f[0])
            response.append(f[0])
            f.pop(0)
            print(f)
            response.append(f)
            print("\n")

        response.append(worf_quote)
        response = json.dumps(response)
        return response, 200

    except Exception as ex:
        response = json.dumps({'response': str(ex)})
        return response, 400


def remove_brackets(text):
    if text:
        text = re.sub('(\(.*?\))|(\[.*?\])', '', str(text))
    else:
        pass
    return text

# def get_large_audio_transcription(path):
#     r = sr.Recognizer()
#     """
#     Splitting the large audio file into chunks
#     and apply speech recognition on each of these chunks
#     """
#     # open the audio file using pydub
#     sound = AudioSegment.from_wav(path)
#     # split audio sound where silence is 700 miliseconds or more and get chunks
#     chunks = split_on_silence(sound,
#                               # experiment with this value for your target audio file
#                               min_silence_len=500,
#                               # adjust this per requirement
#                               silence_thresh=sound.dBFS - 14,
#                               # keep the silence for 1 second, adjustable as well
#                               keep_silence=500,
#                               )
#     folder_name = "audio-chunks"
#     # create a directory to store the audio chunks
#     if not os.path.isdir(folder_name):
#         os.mkdir(folder_name)
#     whole_text = ""
#     # process each chunk
#     for i, audio_chunk in enumerate(chunks, start=1):
#         # export audio chunk and save it in
#         # the `folder_name` directory.
#         chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
#         audio_chunk.export(chunk_filename, format="wav")
#         # recognize the chunk
#         with sr.AudioFile(chunk_filename) as source:
#             audio_listened = r.record(source)
#             # try converting it to text
#             try:
#                 text = r.recognize_google(audio_listened)
#             except sr.UnknownValueError as e:
#                 print("Error:", str(e))
#             else:
#                 text = f"{text.capitalize()}. "
#                 print(chunk_filename, ":", text)
#                 whole_text += text
#     # return the text for all chunks detected
#     return whole_text

# worf_quote  = " Hello everyone, I am sellig the Product which is an office Star Manager chair its type is ergonomic
# chair and color is black. Material is Polyester, Nylon, Foam, Polypropylene, Alloy Steel. Its model is Riley
# Ventilated. Height is adjustable, Speciality is that it is rolling chair with arm set and cushion is available,
# and its brand is powerstone. Price is 150$."
