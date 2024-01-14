from openai import OpenAI

client = OpenAI()

def convertAudioToText(audio_file):
  transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
  )
  text = transcript.text
  return text