# Blind Guide

### Installation and Setup
#### Clone the project using git commands

```Bash
git clone https://github.com/MohamedKhedr07/blind-guide.git
```
```Bash
cd blind-guide/
```
#### Run the API server

   - You should have python poetry installed on your local machine (poetry is python packages manager) [installing tutorial](https://python-poetry.org/docs/)
   - Run the following commands:
       - ```Bash
            cd Backend/
            ```
        If you run the API for the first time after cloning, run the following 2 commands
        - ```Bash
            poetry update

            poetry run python api/manage.py migrate
            ```
        then
        - ```Bash
            poetry run python api/manage.py runserver
            ```
        to start the API server.
