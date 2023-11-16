import tkinter
from tkinter.messagebox import askyesno


def test_user_validation():
    root = tkinter.Tk()
    root.withdraw()

    answer = askyesno(
        title='Confirmation',
        message='Do you approve this results?'
    )

    root.destroy()

    assert answer == True
