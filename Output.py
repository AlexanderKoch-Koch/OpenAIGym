import sys

iter_info = []

def showTrainingProgress(score, scalar, iteration):
    message = ""
    for i in range(0, int(score * scalar)):
        message += "|"
    print(str(iteration) + ": " + message)
    #sys.stdout.flush()

def addIterationInfo(var_name, value):
    iter_info.append(var_name + ": " + str(value))

def showIterationInfo():
    message = ""
    for i in iter_info:
        message += i + "  "
    print(message)
    iter_info.clear()
