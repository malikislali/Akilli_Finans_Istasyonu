import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'quant_web_project.settings')

app = Celery('quant_web_project')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
