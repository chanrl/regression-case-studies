import re

def clean_tire(row):
    if row == 0:
        return 0
    if isinstance(row, int):
        return float(row)
    elif isinstance(row, float):
        return row
    elif row == 'None or Unspecified':
        return 0
    elif isinstance(row, str):
        digits = re.findall('\d+', row)[0]
        return float(digits)
    else:
        return 'otherthing'
        
def clean_under(row):
    if row == 0:
        return 0
    elif row == 'None or Unspecified':
        return 0
    elif isinstance(row, str):
        digits = re.findall('\d+', row)[0]
        return float(digits)
    else:
        return 'otherthing'

def clean_fork(row):
    if row == 'None or Unspecified':
      return 0
    elif row == 'Yes':
      return 1
    else:
      return 0

def clean_blade(row):
    if row == 'None or Unspecified':
      return 0
    elif isinstance(row, str):
      digits = re.findall('\d+', row)[0]
      return float(digits)
    else:
      return 0

def clean_stick(row):
    if row == 'None or Unspecified':
      return 0
    elif isinstance(row, str):
      digits = re.findall('\d+', row)[0]
      return float(digits)
    else:
      return 0

def clean_yesno(row):
    if row == 'None or Unspecified':
      return 0
    elif row == 'No':
      return 0
    elif row == 'Yes':
      return 1
    else:
      return 0
