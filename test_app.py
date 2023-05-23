import requests

resp = requests.post("https://callmodel-lq6mcdg4lq-lm.a.run.app", files={'file': open('test_images/happy-woman-2.jpg', 'rb')})

print(resp.json())