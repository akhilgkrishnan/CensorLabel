path = 'home/sample.mp4'

akhil = []

akhil = path.split(".")
video = akhil[0].split("/")
print(video[len(video)-1])