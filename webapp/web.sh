#!/bin/sh

python manage.py collectstatic --no-input
python manage.py makemigrations
python manage.py migrate --no-input

echo "Running Server..."
gunicorn --workers 3 --bind 0.0.0.0:8000 webapp.wsgi:application