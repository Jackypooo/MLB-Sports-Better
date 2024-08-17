#Modules For Creating Data To Feed Into Model
import statsapi
import datetime

#Modules For Model
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

#GUI
from tkinter import *

#Modules for Saving and Loading Models
import os
import json
 

#Simple Python Script That uses the STATS API MLB API wrapper to generate basic pitching data and uses a simple Convolutional Neural Network to Predict Game Winners
#Default Input Data (Same for away and home teams): Avg Pitcher ERA, Pitcher W/L, Pitcher IP
#Default Output Data: HomeTeam Win Vs AwayTeam Win

class Data:
    '''Class for generating Training and Testing Data'''
    def genTrainData(self, features, team, endDate=datetime.datetime.now().strftime("%m/%d/%Y"), startDate="04/01/2024", numGames = 100, numTeams=5):
        '''Function to generate training data based on certain features modeled around a team'''

        #Returns the id of a team based on name input
        selectedTeam = statsapi.lookup_team(team)[0]["id"]

        #Initialize return variable
        data = []
        
        #Loop through selected team + numTeams after to get a larger training dataset
        for x in range(numTeams):
            #Generate the scheduels for the target team based on this season and get last x games
            sched = statsapi.schedule(start_date=startDate, end_date=endDate, team=selectedTeam-1+x)
            sched = sched[:numGames]

            #Loop through each game of the schedule
            for x in sched:
                #Initialize return variable
                gameData = []

                #Get the pitchers from each game
                homePitcher = x["home_probable_pitcher"]
                awayPitcher = x["away_probable_pitcher"]


                #Get the outcome of each game
                homeScore = x["home_score"]
                awayScore = x["away_score"]

                #Get the ids of the pitchers that pitched in each game
                homePitcherId = (next(x['id'] for x in statsapi.get('sports_players',{'season':datetime.datetime.now().year,'gameType':'W'})['people'] if x['fullName']==homePitcher))
                awayPitcherId = (next(x['id'] for x in statsapi.get('sports_players',{'season':datetime.datetime.now().year,'gameType':'W'})['people'] if x['fullName']==awayPitcher))

                #From the pitchers id get their related stats (Era, etc)
                homePitcherStats = statsapi.player_stats(homePitcherId, group="Pitching", type="Season")
                awayPitcherStats = statsapi.player_stats(awayPitcherId, group="Pitching", type="Season")

                #Based on the features we want for our model loop through and add said stats from the pitchers of each game to the data return variable
                for x in features:
                    gameData.append(findInString(x, homePitcherStats))
                    gameData.append(findInString(x, awayPitcherStats))
                
                #Add game outcome to the data
                gameData.append(1 if (homeScore > awayScore) else 0)
                data.append(gameData)
        
        return data
    
    def genRunData(self, features, teamInput):
        '''Generates data required to run model on specific team'''
        #Use nym by default
        teamInput = "nym" if teamInput == "" else teamInput
        print(f"Generating data for {teamInput} game on {selectedDate}")

        #Find the teams id and initialize data return variable
        teamId = statsapi.lookup_team(teamInput)[0]["id"]
        data = []

        #Get the game data for the upcoming game
        gameData = statsapi.schedule(start_date=selectedDate, end_date=selectedDate, team=teamId)

        #If team isnt playing on said date or another error occurs then return error message
        if gameData == []:
            print(f"Game not found. Maybe {teamInput}'s aren't playing on {selectedDate} or the game is too far into the future")
            return
        
        #Get the probable pitcher of the upcoming game along with their id
        homeProbablePitcher = gameData[0]["home_probable_pitcher"]
        awayProbablePitcher = gameData[0]["away_probable_pitcher"]

        #Get pitchers Ids
        try:
            homePitcherId = (next(x['id'] for x in statsapi.get('sports_players',{'season':datetime.datetime.now().year,'gameType':'W'})['people'] if x['fullName']==homeProbablePitcher))
            awayPitcherId = (next(x['id'] for x in statsapi.get('sports_players',{'season':datetime.datetime.now().year,'gameType':'W'})['people'] if x['fullName']==awayProbablePitcher))
        except:
            print("Error getting pitchers IDs")
            return

        #If probable pitchers are not confirmed throw error
        if homeProbablePitcher == "" or awayProbablePitcher == "":
            print("Error Probable Pitchers Not Available")
            return
        
        #Get the pitchers stats and append them to the data variable
        homePitcherStats = statsapi.player_stats(homePitcherId, group="Pitching", type="Season")
        awayPitcherStats = statsapi.player_stats(awayPitcherId, group="Pitching", type="Season")

        for x in features:
            data.append(findInString(x, homePitcherStats))
            data.append(findInString(x, awayPitcherStats))

        #Return the pitchers stats
        print("Data Gen Success!")
        return data, gameData

class Model(nn.Module):
    '''Generates Neural Network Model'''
    def __init__(self, inputs, layer1Size=8, layer2Size=8, outputs=2):
        '''Neural Network initialization based on layers and their sizes'''
        super().__init__()

        self.hidden1 = nn.Linear(inputs, layer1Size)
        self.hidden2 = nn.Linear(layer1Size, layer2Size)
        self.out = nn.Linear(layer2Size, outputs)

    def forward(self, x):
        '''Forward method to pass forward the data using relu activation functions'''
        x = nn.functional.relu(self.hidden1(x))
        x = nn.functional.relu(self.hidden2(x))
        x = self.out(x)
        return x

def getModelAttributes(event):
    '''Function that gets the attributes for model creation'''
    d = Data()
    print("Getting Model Attributes")

    epochs = 100 if epochsEntry.get() == "" else int(epochsEntry.get())
    print(f'Using {epochs} Epochs')


    inputFeatures = [possibleInputFeatures[x[0]] for x in enumerate(CBVariables) if x[1].get() == 1]
    print(f'Using {inputFeatures}')

    name = "Unnamed Model" if modelNameEntry.get() == "" else modelNameEntry.get()

    team = "nym" if teamTrainEntry.get() == "" else teamTrainEntry.get()
    print(f'Using {team}')

    if(len(inputFeatures) < 1): return

    #Generates training data based on features team and numgames specified
    data = d.genTrainData(features=inputFeatures, team=team, numgames=250)
    print("Successfull Data Extraction")

    #Calls genmodel function to generate a model
    genModel(data, inputFeatures, name, epochs)

def genModel(d, modelInputFeatures, name, epochs=100):
    '''Generates and Saves Model Based on Features Gathered from GUI'''
    print("Generating Model")
    model = Model(inputs=len(modelInputFeatures)*2)

    X = []
    y = []

    #Initializes input variables and prediction variables
    for x in d:
        outputValues = x.pop()
        X.append(x)
        y.append(outputValues)


    X = np.array(X)
    y = np.array(y)

    #Splits data into training and testing variables
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)

    xTrain = torch.FloatTensor(xTrain)
    xTest = torch.FloatTensor(xTest)
    yTrain = torch.LongTensor(yTrain)
    yTest = torch.LongTensor(yTest)

    #Initializes our loss function and optimizer
    lossFunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #Trains the Model
    print("Training Model:")
    for x in range(epochs):
        yPred = model.forward(xTrain)
        loss = lossFunc(yPred, yTrain)

        #Prints output every 10 epochs
        if x % 10 == 0: print(f'Epoch: {x}, Loss: {loss}')

        #Back propigation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Done Training!")
    print("Testing")

    #Tests Model
    with torch.no_grad():
        yEval = model.forward(xTest)
        testLoss = lossFunc(yEval, yTest)
    
    #Prints Output
    print("Done!")
    print(f'Training Loss: {loss}, Testing Loss: {testLoss}')
    
    #Saves Model as.pt file
    saveas = f"Models/{name}.pt"
    torch.save(model.state_dict(), saveas)

    #Saves Info Relating to Model
    modelInfo = {
        "Model Input Features" : modelInputFeatures
    }
    
    with open(f"Models/{name}.json", 'w') as outfile:
        json.dump(modelInfo, outfile)

def findInString(x, inString):
    '''Returns Corresponding Value Based On String'''
    #Examples if searching for b in "a:45, b:32, c:0" would return 32
    y = ""

    endFound = False
    count = 0

    #Check in string for value and if found check for numbers after it
    while endFound != True:
        location = inString.find(x) + len(x) + 2 + count
        if(inString[location].isnumeric() or inString[location] == "."):
            y += inString[location]
        else:
            endFound = True

        count += 1
    if y == '' or y == ".": y=0

    return float(y)

def updateOutputData(update):
    '''When model has output post it in the output panel'''
    if update != []:
        output = update[0]
        gameData = update[1]

        successLabel = Label(master=outputFrame, text="Success!", fg='Green', bg="Light Gray")
        successLabel.place(x=50,y=5)

        dataLabel = Label(master=outputFrame, text=f"{output}", bg="Light Gray")
        dataLabel.place(x=35, y=30)

        winningTeam = gameData["home_name"] if output[1] > output[0] else gameData["away_name"]

        predictionLabel = Label(master=outputFrame, text=f"Pred: {winningTeam}", bg="Light Gray", justify=CENTER)
        predictionLabel.place(x=10, y=55)

        #Create confidence value via distance formula
        conf = round(np.abs(output[0]-output[1])*100,1)
        confidenceLabel = Label(master=outputFrame, text=f"Confidence: {conf}", bg="Light Gray", justify=CENTER)
        confidenceLabel.place(x=20, y=80)
    else:
        outputFrame.place(x=5, y=185)

def loadModel(event):
    '''Loads a saved .pt model from file'''
    d = Data()

    #Clears Output Field
    updateOutputData([])

    #Check If Model File Exists:
    modelName = "Unnamed Model" if modelUseEntry.get() == "" else modelUseEntry.get()

    #Gets path of both files
    modelPath = os.getcwd() + f"/Models/{modelName}.pt"
    infoPath = os.getcwd() + f"/Models/{modelName}.json"

    #Stops if Cannot find Model or Info Path
    if not (os.path.exists(modelPath) and os.path.exists(infoPath)):
        print("Either info file or model file does not exist.  Please double check your files and try again")
        return
    
    #Check if Info File Exists
    os.path.exists(os.getcwd() + "/Models/Unnamed Model.pt")

    #Loades Saved Model
    savedModel = torch.load(modelPath)
    savedModelInputLayerSize = len(savedModel["hidden1.weight"][0])

    newModel = Model(savedModelInputLayerSize)
    newModel.load_state_dict(savedModel)

    #Opens info file
    with open(infoPath, 'r') as openfile:
        modelInfo = json.load(openfile)["Model Input Features"]

    #Generates Test Data
    data = d.genRunData(modelInfo, teamRunEntry.get())

    #If generation successfull run the saved model with the generated training data
    if data != None: runModel(newModel, data)

def runModel(model, data):
    '''Runs loaded model'''
    runData = torch.FloatTensor(data[0])
    gameData = data[1][0]

    #Runs Model
    output = model.forward(runData)
    output = output.detach().numpy().round(3)
    
    #Updates data after model ran
    updateData = [output, gameData]
    updateOutputData(updateData)

def changeDate(event):
    '''Changes the date'''
    dateWindow = Tk()
    dateWindow.title("Date")

    dateWindow.geometry("60x110")

    # Constrains Window
    dateWindow.minsize(60, 110)
    dateWindow.maxsize(60, 110)

    #Gets month day and year from user
    monthLabel = Label(master=dateWindow, text="Month: ")
    monthLabel.place(x=0, y=0)

    monthEntry = Entry(master=dateWindow, width=8)
    monthEntry.insert(END, datetime.datetime.now().month)
    monthEntry.place(x=50,y=0)


    dayLabel = Label(master=dateWindow, text="Day: ")
    dayLabel.place(x=0, y=25)
    
    dayEntry = Entry(master=dateWindow, width=8)
    dayEntry.insert(END, datetime.datetime.now().day)
    dayEntry.place(x=50,y=25)



    yearLabel = Label(master=dateWindow, text="Year: ")
    yearLabel.place(x=0, y=50)

    yearEntry = Entry(master=dateWindow, width=8)
    yearEntry.insert(END, datetime.datetime.now().year)
    yearEntry.place(x=50,y=50)

    def save(event):
        '''Saves Date to variable'''
        saveDate = f'{yearEntry.get()}-{monthEntry.get()}-{dayEntry.get()}'

        try:
            saveDate = datetime.datetime.strptime(saveDate, "%Y-%m-%d").date()
        except:
            print("Provided date format unaccepted")
            return

        global selectedDate
        selectedDate = saveDate


        #Updates Date Button
        dateButton = Button(master=runFrame, text=saveDate)
        dateButton.place(x=50,y=75)
        dateButton.bind("<Button-1>", changeDate)

        dateWindow.destroy()

    saveButton = Button(master=dateWindow, text="Save")
    saveButton.place(x=50,y=75)
    saveButton.bind("<Button-1>", save)

#General Settings:
selectedDate = datetime.datetime.now().date()

#Window Instantiation and General Settings
window = Tk()
window.title("MLB Game Predictor :)")

window.geometry("400x325")

# Constrains Window
window.minsize(400, 325)
window.maxsize(400, 325)

#Right Hand (Train) Side
trainFrame = Frame()

#Title of Frame
trainFrameTitle = Label(master=trainFrame, text="Model Generator")
trainFrameTitle.pack()

#Generates CheckBoxes Based on the possible input features
possibleInputFeatures = ["strikeOuts","era","inningsPitched", "whip", "winPercentage", "homeRuns"]
CBs = []
CBVariables = []

#Creates Variables And Checkbox Objects and PLaces Them
for i, x in enumerate(possibleInputFeatures):
    var = IntVar()
    CBs.append(Checkbutton(master=trainFrame, text=x, variable=var, onvalue=1, offvalue=0))
    CBVariables.append(var)
    CBs[i].place(x=0, y=25*(i+1))

#Attributes for the "Generate Model Button"
genButton = Button(text="Generate Model", width=15, height=1)
genButton.place(y=275, x=50)
genButton.bind("<Button-1>", getModelAttributes)


modelNameLabel = Label(master=trainFrame, text="Model Name: ")
modelNameLabel.place(y=(len(possibleInputFeatures) + 1)*25, x=0)

modelNameEntry = Entry()
modelNameEntry.place(y=(len(possibleInputFeatures) + 1)*25,x=85)

teamTrainLabel = Label(master=trainFrame, text="Training Team: ")
teamTrainLabel.place(y=(len(possibleInputFeatures) + 2)*25, x=0)

teamTrainEntry = Entry()
teamTrainEntry.place(y=(len(possibleInputFeatures) + 2)*25,x=85)

epochsLabel = Label(master=trainFrame, text="Epochs: ")
epochsLabel.place(y=(len(possibleInputFeatures) + 4)*25, x=0)

epochsEntry = Entry(master=trainFrame)
epochsEntry.place(y=(len(possibleInputFeatures) + 4)*25,x=85)   

#Left Hand Side 
runFrame = Frame()

runFrameTitle = Label(master=runFrame, text="Test:")
runFrameTitle.pack()

modelUseLabel = Label(master=runFrame, text="Model: ")
modelUseLabel.place(y=25, x=0)

modelUseEntry = Entry(master=runFrame, width=18)
modelUseEntry.place(y=25, x=50)

teamRunLabel = Label(master=runFrame, text="Team: ")
teamRunLabel.place(x=0,y=50)

teamRunEntry = Entry(master=runFrame, width=18)
teamRunEntry.place(x=50,y=50)

dateLabel = Label(master=runFrame, text="Date: ")
dateLabel.place(x=0,y=75)

dateButton = Button(master=runFrame, text=selectedDate)
dateButton.place(x=50,y=75)
dateButton.bind("<Button-1>", changeDate)

runModelButton = Button(master=runFrame, text="Run Model", width=15, height=1)
runModelButton.place(y=125, x=25)
runModelButton.bind("<Button-1>", loadModel)

#Button to serve as background for output
outputLabel = Label(master=runFrame, text="Output: ")
outputLabel.place(x=60,y=160)

outputFrame = Button(master=runFrame, bg="Light Gray", text="", width=20, height=8,state=DISABLED)
outputFrame.place(x=5, y=185)

#Packs Both Sides of the model to there corrisponding places on the window
trainFrame.pack(side=LEFT,fill=BOTH, expand=True)
runFrame.pack(side=RIGHT,fill=BOTH, expand=True)


mainloop()