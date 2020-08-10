# Generated by Django 3.0.5 on 2020-04-23 10:46

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Faces',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('date', models.CharField(default='1399/02/04', editable=False, max_length=10)),
                ('photo', models.FileField(blank=True, upload_to='media')),
            ],
        ),
    ]