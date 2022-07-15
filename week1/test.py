class Dog():
    species="Canis"

    def __init__(self,name,age):
        self.age = age
        self.name = name
    
    def speak(self,sound):
        return f"The dog {self.name} makes the sound {self.sound}"

captain=Dog("Captain", 14)
captain.name
captain.age


