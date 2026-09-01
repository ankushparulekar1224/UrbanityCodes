import sys
import easygui

# 1. Display a welcome message box
easygui.msgbox(
    msg="Welcome to the EasyGUI Demo Application!", 
    title="Introduction", 
    ok_button="Let's Start"
)

# 2. Get a text input from the user
user_name = easygui.enterbox(
    msg="Please enter your name:", 
    title="User Information"
)

# Handle cases where the user cancels or closes the dialog box
if user_name is None:
    easygui.msgbox("Operation cancelled by the user.", "Exit")
    sys.exit()

# 3. Present a list of choices using a choice box
msg = f"Hello {user_name}! What is your favorite programming language?"
title = "Language Survey"
choices = ["Python", "JavaScript", "C++", "Java", "Rust"]

user_choice = easygui.choicebox(msg, title, choices)

# 4. Display the final result based on user selections
if user_choice:
    easygui.msgbox(
        msg=f"Great choice, {user_name}! {user_choice} is awesome.", 
        title="Survey Result"
    )
else:
    easygui.msgbox(
        msg=f"You didn't select a language, {user_name}.", 
        title="Survey Result"
    )