import os
import sys
import django

# Setup Django ORM context inside cvat_server
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cvat.settings.development")
django.setup()

from cvat.apps.webhooks.models import Webhook, WebhookTypeChoice
from cvat.apps.engine.models import Project

def setup_auto_trainer_webhook():
    target_url = "http://host.docker.internal:5000"
    
    # Official CVAT Webhook Event Keys
    valid_events = [
        "update:job",
        "update:task",
        "create:job",
        "create:task"
    ]
    
    project = Project.objects.first()
    if not project:
        print("❌ No CVAT Project found in database.")
        return

    # Delete any duplicate or outdated webhooks pointing to port 5000
    Webhook.objects.filter(target_url=target_url).delete()

    # Create active webhook bound to Project 1
    webhook = Webhook.objects.create(
        target_url=target_url,
        description="Auto Fine-Tuning Receiver for Thermal YOLOv8",
        type=WebhookTypeChoice.PROJECT,
        project=project,
        events=valid_events,
        is_active=True
    )
    
    print(f"✅ Webhook successfully bound to CVAT Database!")
    print(f"🆔 Webhook ID: {webhook.id}")
    print(f"🎯 Target URL: {webhook.target_url}")
    print(f"⚡ Bound Events: {webhook.events}")
    print(f"📁 Project ID: {project.id} ({project.name})")

if __name__ == "__main__":
    setup_auto_trainer_webhook()
