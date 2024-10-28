# Generated by Django 5.0.4 on 2024-04-20 22:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('GDSC', '0002_item_delete_todo'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='item',
            name='description',
        ),
        migrations.RemoveField(
            model_name='item',
            name='name',
        ),
        migrations.AddField(
            model_name='item',
            name='age',
            field=models.IntegerField(default=18),
        ),
        migrations.AddField(
            model_name='item',
            name='methodPreferences',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='item',
            name='school',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='item',
            name='studyDescription',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='item',
            name='studyGoal',
            field=models.TextField(default=''),
        ),
    ]