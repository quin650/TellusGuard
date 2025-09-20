import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEBUG = os.getenv("DEBUG", "False")

EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = "smtp.gmail.com"
EMAIL_FROM = os.getenv("EMAIL_FROM", default="")  
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER", default="")
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD", default="")
EMAIL_PORT = 587
EMAIL_USE_TLS = True

DATABASES_LOCAL = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "tellused",
        "USER": "postgres",
        "PASSWORD": os.getenv("pgpassword", ""),
        "HOST": "localhost",
    }
}