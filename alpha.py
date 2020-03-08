import tkinter
from tkinter import filedialog
import os
import tkinter.messagebox
import subprocess

#initilization of main window
m=tkinter.Tk()
m.title('Statutory warning generator') 
m.geometry("267x180")
m.configure(background="light blue")

#getting file location
m.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("mp4 files","*.mp4"),("mpv files","*.mpv"),("all files","*.*")))
print (m.filename)
tempfilename = (m.filename[:15] + '..') if len(m.filename) > 15 else m.filename

#start button Call
def startCall():
    tkinter.messagebox.showinfo( "SWG", "Press ok to confirm")
    print(m.filename)
    
    #to covert video to frame
    os.system('ffmpeg -i '+m.filename+' -vf fps=1 %d.jpg -hide_banner')
    
    #to get video time in second
    os.system('ffmpeg -i '+m.filename+' 2>&1 | grep \"Duration\"| cut -d \' \' -f 4 | sed s/,// | sed \'s@\..*@@g\' | awk \'{ split($1, A, \":\"); split(A[3], B, \".\"); print 3600*A[1] + 60*A[2] + B[1] }\'')
    output = subprocess.check_output("ffmpeg -i "+m.filename+" 2>&1 | grep \"Duration\"| cut -d ' ' -f 4 | sed s/,// | sed 's@\..*@@g' | awk '{ split($1, A, \":\"); split(A[3], B, \".\"); print 3600*A[1] + 60*A[2] + B[1] }'", shell=True);
    
    #to convert byte to string
    li = str( output )[2:-3]
    noofframe = int(li)
    print(noofframe)
    
    #to create test file for extracted frame
    f= open("../darknet/data/valid.txt","w")
    for i in range(noofframe):
        f.write("../alpha/%d.jpg\n" % (i+1))
    f.close()
    
    #to test all frame in darknet
    os.chdir('../darknet')
    os.system('./darknet detector valid data/obj.data y.cfg backup/yolov3_16000.weights')
    
    #to get file directory and name
    nfilename=m.filename
    li = list(nfilename.split("/"))  
    filename = li[-1]  
    newfiledir = nfilename.replace(filename,'')

    

    #to create srt from result file
    li = list(filename.split("."))
    newfilename = li[0]
    print(newfilename)
    print(newfiledir)
    i=1
    overlay=0
    f2= open(newfiledir+newfilename+".srt","w+")
    f1= open("../darknet/results/comp4_det_test_smoke.txt","r+")
    for line in f1:
        li = list(line.split(" ")) 
        time=int(li[0])    
        per=float(li[1])   
        if per > 0.30 and overlay!=time:

            #to convert second to hour format
            mi, s = divmod(time, 60)
            h, mi = divmod(mi, 60)
            
            #to create srt file
            f2.write("%d\n" % (i))
            f2.write("%d:" % (h))
            if s!=0:
                f2.write("%d:" % (mi))
                f2.write("%d,000 --> " % (s-1))
            else:
                f2.write("%d:" % (mi-1))
                f2.write("59,000 --> ")
            f2.write("%d:" % (h))
            if s!=59:
                f2.write("%d:" % (mi))
                f2.write("%d,000\n"% (s))
            else:
                f2.write("%d:" % (mi+1))
                f2.write("00,000\n")
            f2.write("smoking is injurious to health\n\n")
            i=i+1
            overlay=time
            print(overlay)


        print(li)
    f1.close()
    f2.close()
    os.chdir('../alpha')
    for i in range(1,noofframe+1):
        os.remove(str(i)+".jpg")
    

#initilization of button and labels
button1 = tkinter.Button(m, text='Start', width=15, command=startCall) 
button2 = tkinter.Button(m, text='Exit', width=15, command=m.destroy) 
label1 = tkinter.Label(m, text='Filename : ')
label2 = tkinter.Label(m, text=tempfilename)
label3 = tkinter.Label(m, text='')
label4 = tkinter.Label(m, text='')
label5 = tkinter.Label(m, text='')
label6 = tkinter.Label(m, text='')
label7 = tkinter.Label(m, text='')
label8 = tkinter.Label(m, text='')
label6.grid(row=0, column=1)
label7.grid(row=1, column=1)
label8.grid(row=2, column=1)
label1.grid(row=3, column=0)
label2.grid(row=3, column=1)
label3.grid(row=4, column=1)
label4.grid(row=5, column=1)
label5.grid(row=6, column=1)
button1.grid(row=7, column=0)
button2.grid(row=7, column=1)
 
#to run window until event
m.mainloop()
