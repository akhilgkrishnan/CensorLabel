from tkinter import *
from tkinter import filedialog


root = Tk()
root.geometry("1600x900+0+0")
root.title("Statutory Generating System")

def selectFile():
    inputFileName =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("mp4 files","*.mp4"),("all files","*.*")))

c = Canvas(root, bg="blue", height=300, width=300)
background_image=PhotoImage(file ='Images/tkbg.png')
background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
Tops = Frame(root,bg="white",width = 1600,height=50,relief=SUNKEN)
Tops.pack(side=TOP)
#-----------------INFO TOP------------
lblinfo = Label(Tops, font=( 'aria' ,30, 'bold' ),text="Statutory Warning Generator",fg="steel blue",bd=10,anchor='w')
lblinfo.grid(row=0,column=0)

ButtonFrame = Frame(root,width = 1600,height = 500, bg="#afdbb7")
ButtonFrame.pack()

#Read input video
fileButton = Button(ButtonFrame,text= "InputVideo",command=selectFile)
fileEntry = Entry(ButtonFrame)
fileEntry.grid(row=0,column=0)
fileButton.grid(row=0,column=1)

#Select the movie language
langLabel = Label(ButtonFrame,text="Language :")

variable = StringVar(ButtonFrame)
variable.set("Malayalam")
langOption = OptionMenu(ButtonFrame,variable,"Malayalam","English")

langLabel.grid(row=0,column=3)
langOption.grid(row=0,column=4)

footer= Frame(root)
footer.pack(side=BOTTOM)
startButton = Button(footer,text="Start")
exitButton = Button(footer,text="Exit",command=root.quit)
startButton.grid(row=0,column=0)
exitButton.grid(row=0,column=1)







c.pack()
root.mainloop()