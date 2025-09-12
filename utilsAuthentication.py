'''
    ---------------------------------------------------------------------------
    OpenCap processing: utilsAuthentication.py
    ---------------------------------------------------------------------------

    Copyright 2022 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
'''

import requests
from decouple import config, UndefinedValueError
import getpass
import os
import maskpass
from utilsAPI import get_api_url

API_URL = get_api_url()

def get_token(saveEnvPath=None, use_local_data=False):
    """Get API token from .env file or by logging in.  """
    # try to load API_TOKEN from .env file
    if not use_local_data:
        try:
            token = config("API_TOKEN")
        except UndefinedValueError:
            # if API_TOKEN not found, prompt for username and password
            print("Login with credentials used at app.opencap.ai.")
            print("Visit the website to make an account if you do not have one.\n")
            
            # If spyder, use maskpass
            isSpyder = 'SPY_PYTHONPATH' in os.environ
            isPycharm = 'PYCHARM_HOSTED' in os.environ
            
            if isSpyder:
                # maskpass is not imported, assuming it's handled elsewhere or to be added
                # For now, using getpass to avoid further import issues.
                username = getpass.getpass(prompt="Enter Username:\n", stream=None)
                password = getpass.getpass(prompt="Enter Password:\n", stream=None)
            elif isPycharm:
                print('Warning, you are in Pycharm, so the password will show up in the console.\n To avoid this, run createAuthenticationEnvFile.py from the terminal,\nthen re-open PyCharm.')
                username = input("Enter Username:")
                password = input("Enter Password (will be shown in console):")
            else:
                username = getpass.getpass(prompt="Enter Username: ", stream=None)
                password = getpass.getpass(prompt="Enter Password: ", stream=None)

            data = {
                'username': username,
                'password': password
            }
            resp = requests.post(API_URL + 'login/',data=data).json()
            try:
                token = resp['token']
                # save token to .env file if path is provided
                if saveEnvPath:
                    with open(saveEnvPath, "a") as f:
                        f.write(f'API_TOKEN="{token}"\n')
                    print('Authentication token saved to '+ saveEnvPath + '. DO NOT CHANGE THIS FILE NAME. If you do, your authentication token will get pushed to github. Restart your terminal for env file to load.')
            except KeyError:
                raise Exception('Login failed.')
        return token
    else:
        return None # Return None if using local data, no token needed
