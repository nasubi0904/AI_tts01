import requests

def get_speakers():
    url = "http://localhost:50021/speakers"  # VOICEVOX APIのエンドポイント
    response = requests.get(url)

    if response.status_code == 200:
        speakers = response.json()
        for speaker in speakers:
            name = speaker['name']
            style_names = [style['name'] for style in speaker['styles']]
            style_ids = [style['id'] for style in speaker['styles']]
            for style_id, style_name in zip(style_ids, style_names):
                print(f"Speaker: {name}, {style_name} id: {style_id}")
            
    else:
        print(f"Error: {response.status_code}")

if __name__ == "__main__":
    get_speakers()