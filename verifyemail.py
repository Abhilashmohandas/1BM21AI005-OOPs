import re

def is_valid_email(email):
  pattern = r'^[\w\.-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$'

  if re.match(pattern, email):
        return True
  else:
        return False

email_address = "email@email.com"
result = is_valid_email(email_address)

if result:
    print(f"The email '{email_address}' is valid.")
else:
    print(f"The email '{email_address}' is not valid.")
