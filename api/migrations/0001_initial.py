# Generated by Django 2.0.5 on 2018-05-13 14:39

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Fathernames',
            fields=[
                ('fathername', models.TextField(primary_key=True, serialize=False)),
            ],
        ),
        migrations.CreateModel(
            name='Lastnames',
            fields=[
                ('lastname', models.TextField(primary_key=True, serialize=False)),
            ],
        ),
        migrations.CreateModel(
            name='Names',
            fields=[
                ('name', models.TextField(primary_key=True, serialize=False)),
            ],
        ),
    ]
