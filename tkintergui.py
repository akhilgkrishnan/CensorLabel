from tkinter import *
from tkinter import filedialog


root = Tk()
root.geometry("1600x700+0+0")
root.title("Statutory Generating System")

c = Canvas(root, bg="blue", height=300, width=300)
background_image=PhotoImage(file ='Images/tkbg.png')
background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
Tops = Frame(root,bg="white",width = 1600,height=50,relief=SUNKEN)
Tops.pack(side=TOP)
#-----------------INFO TOP------------
lblinfo = Label(Tops, font=( 'aria' ,30, 'bold' ),text="Statutory Warning Generator",fg="steel blue",bd=10,anchor='w')
lblinfo.grid(row=0,column=0)

root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("mp4 files","*.mp4"),("all files","*.*")))





c.pack()
root.mainloop()