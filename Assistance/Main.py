FileDirectory = "F:/BAR Beyond All Reason/Beyond-All-Reason/data"

while(True):
    print("Units available to command: \n")
    with open(FileDirectory + "/currentUnits.txt", "r") as file:
        content = file.read()
        print(content)

    command = input("Please enter command: ")
    if command == "quit":
        break
    elif command == "skip":
        print("Loading current data")
    elif "Move" in command:
        with open(FileDirectory+"/unitCommands.txt", "w") as file:
            commandParts = command.split()
            file.write("Command: MoveUnit " + commandParts[1] + " " + commandParts[2] + " " + commandParts[3] + " "
                       + commandParts[4] + "\n")
    else:
        print("No valid command")

