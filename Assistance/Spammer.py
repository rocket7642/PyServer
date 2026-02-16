import random
import time

FileDirectory = "F:/BAR Beyond All Reason/Beyond-All-Reason/data"
loops = 0

while True:
    with open(FileDirectory + "/currentUnits.txt", "a") as file:
        file.write(
            "Fully Large Message, lets test the transfer and see what sticks, maybe this works, maybe it doesn't")
    # time.sleep(0.1)
