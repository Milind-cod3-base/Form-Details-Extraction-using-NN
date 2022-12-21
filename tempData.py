def save(path):
    with open("filename.txt","w") as file:
        file.write(path)


def read():
    with open("filename.txt","r") as file:
        return file.read()