import time
import subprocess
import requests

def test_endpoints():
    print('Starting server...')
    proc = subprocess.Popen(['python', 'app.py'])
    time.sleep(5)

    print('\nTesting endpoints...')
    endpoints = [
        ('GET', 'http://127.0.0.1:5000/api/train/status', None),
        ('POST', 'http://127.0.0.1:5000/api/chat', {'message': 'test', 'session_id': '123'}),
        ('POST', 'http://127.0.0.1:5000/api/math_step', {'word': 'test'}),
        ('POST', 'http://127.0.0.1:5000/api/visualize', {'script': 'visualize_architecture', 'args': []}),
        ('POST', 'http://127.0.0.1:5000/api/visualize', {'script': 'visualize_attention', 'args': []}),
        ('POST', 'http://127.0.0.1:5000/api/visualize', {'script': 'visualize_embeddings', 'args': []}),
        ('POST', 'http://127.0.0.1:5000/api/visualize', {'script': 'export_onnx', 'args': []})
    ]

    success = True
    for method, url, json_data in endpoints:
        try:
            r = requests.request(method, url, json=json_data)
            print(f'{method} {url} {"(script: " + json_data.get("script") + ")" if json_data and "script" in json_data else ""} -> Status: {r.status_code}')
            if r.status_code != 200:
                success = False
                print("Response:", r.text)
        except Exception as e:
            print(f'Failed {method} {url}: {e}')
            success = False

    print("\nTerminating server...")
    proc.terminate()
    proc.wait()
    assert success, "One or more endpoints failed"
