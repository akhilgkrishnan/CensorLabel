import tkinter.scrolledtext as tkscrolled
import tkinter as tk

default_text = '1234'
width, height = 20,10
TKScrollTXT = tkscrolled.ScrolledText(10, width=width, height=height, wrap='word')

# set default text if desired
TKScrollTXT.insert(1.0, default_text)
TKScrollTXT.pack(side=tk.LEFT)
