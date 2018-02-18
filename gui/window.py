from Tkinter import *
from collections import OrderedDict
from ttk import *

DropdownMenu = OptionMenu


class LayerChoices(object):
    def __init__(self):
        self.layer_choices = OrderedDict()

    def add_choices(self, name, choices):
        self.layer_choices[name] = choices

    def get_choices(self):
        for k, v in self.layer_choices.iteritems():
            yield k, v


class SettingWindow(object):
    def __init__(self,
                 layer_choices,
                 button_onclick,
                 dropdown_callback,
                 entry_callback):
        self.root = Tk()
        self.root.title('Visualise layer options')
        self.mainframe = Frame(self.root)
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)
        self.mainframe.pack(pady=100, padx=100)
        self.entry_callback = entry_callback

        self.setup_dropdrop_menu(layer_choices, dropdown_callback)
        self.setup_entries()
        self.setup_buttons(button_onclick)

    def mainloop(self):
        self.root.mainloop()

    def button_onclick_wrapper(self, button_onclick):
        self.execute_entry_callback()
        return button_onclick

    def execute_entry_callback(self):
        learning_rate = float(self.learning_rate_var.get())
        self.entry_callback('learning_rate', learning_rate)

        batch_size = int(self.batch_size_var.get())
        self.entry_callback('batch_size', batch_size)

    def add_empty_space(self, row, column):
        empty_space = Label(self.mainframe)
        empty_space.grid(row=row, column=column)

    def setup_entries(self):
        self.create_label(7, 1, 'learning rate')
        self.learning_rate_var = StringVar(self.root, name='learning_rate_var')
        learning_rate_entry = Entry(self.mainframe, textvariable=self.learning_rate_var)
        learning_rate_entry.grid(row=7, column=2, padx=5, pady=1)

        self.create_label(8, 1, 'batch size')
        self.batch_size_var = StringVar(self.root, name='batch_size_var')
        batch_size_entry = Entry(self.mainframe, textvariable=self.batch_size_var)
        batch_size_entry.grid(row=8, column=2, padx=5, pady=1)

    def setup_buttons(self, button_onclick):
        button_width = 15

        train_one_epoch_button = Button(self.mainframe,
                                        text='Train one epoch',
                                        command=lambda: self.button_onclick_wrapper(button_onclick)('epoch'))
        train_one_epoch_button.config(width=button_width)
        train_one_epoch_button.grid(row=10, column=2, padx=1, pady=2)

        train_one_batch_button = Button(self.mainframe,
                                        text='Train one batch',
                                        command=lambda: self.button_onclick_wrapper(button_onclick)('batch'))
        train_one_batch_button.config(width=button_width)
        train_one_batch_button.grid(row=11, column=2, padx=1, pady=2)

        reset_button = Button(self.mainframe,
                              text='Reset model weights',
                              command=lambda: button_onclick('reset'))
        reset_button.config(width=button_width)
        reset_button.grid(row=12, column=2, padx=1, pady=2)

        force_stop_button = Button(self.mainframe,
                                   text='Stop training',
                                   command=lambda: button_onclick('force_stop'))
        force_stop_button.config(width=button_width)
        force_stop_button.grid(row=13, column=2, padx=1, pady=2)

    def setup_dropdrop_menu(self, layer_choices, dropdown_callback):
        self.dropdownvars = OrderedDict()

        for i, (name, choices) in enumerate(layer_choices.get_choices()):
            self.setup_dropdown_onchange(i, name, choices, dropdown_callback)

    def setup_dropdown_onchange(self, i, name, choices, dropdown_callback):
        self.dropdownvars[name] = StringVar(self.root, name='strvar_' + name)

        self.create_label(1 + i, 1, name)
        self.create_dropdown_menu(1 + i, 2, name, choices, dropdown_callback)

    def create_label(self, row, column, text):
        label = Label(self.mainframe, text=text)
        label.grid(row=row, column=column)

    def create_dropdown_menu(self, row, column, name, choices, onchange_callback):
        def dropdown_onchange(*args):
            strvar_name = args[0][len('strvar_'):]
            selected_choice = self.dropdownvars[strvar_name].get()

            if onchange_callback is not None:
                onchange_callback(strvar_name, selected_choice)

        dropdownmenu = DropdownMenu(self.mainframe, self.dropdownvars[name], choices[0], *choices)
        dropdownmenu.grid(row=row, column=column, padx=2, pady=1)

        self.dropdownvars[name].trace('w', dropdown_onchange)
#
#
# layer_choices = LayerChoices()
# layer_choices.add_choices('ghd', ['1'])
# layer_choices.add_choices('bn', ['1'])
# layer_choices.add_choices('naive', ['1'])
#
#
# def button(setting_window, x):
#     setting_window.execute_entry_callback()
#     print(x)
#
#
# def entry(y, x):
#     print(y, x)
#
#
# sw = SettingWindow(layer_choices, button, None, entry)
# sw.mainloop()
