import os
import pickle
import PySimpleGUI as sg
from typing import Dict
sg.ChangeLookAndFeel('Black')

layout: list = [
            [sg.Text('Search Term', size=(11,1)), 
             sg.Input(size=(40,1), focus=True, key="TERM"), 
             sg.Radio('Contains', size=(10,1), group_id='choice', key="CONTAINS", default=True), 
             sg.Radio('StartsWith', size=(10,1), group_id='choice', key="STARTSWITH"), 
             sg.Radio('EndsWith', size=(10,1), group_id='choice', key="ENDSWITH")],
            [sg.Text('Root Path', size=(11,1)), 
             sg.Input('/..', size=(40,1), key="PATH"), 
             sg.FolderBrowse('Browse', size=(10,1)), 
             sg.Button('Re-Index', size=(10,1), key="_INDEX_"), 
             sg.Button('Search', size=(10,1), bind_return_key=True, key="_SEARCH_")],
            [sg.Output(size=(50,30)),sg.Output(size=(50,30))]]
window= object = sg.Window('File Search Engine', self.layout, element_justification='left')



       