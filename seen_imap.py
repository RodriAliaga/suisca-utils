import imaplib

IMAP_SERVER = 'IMAP.servidor-correo.net'  # Ej: 'imap.gmail.com'
EMAIL = 'rfq.hue@suisca.com'
PASSWORD = '=3s{dr3jKD_34BaA'

def marcar_todos_como_vistos():
    with imaplib.IMAP4_SSL(IMAP_SERVER) as mail:
        mail.login(EMAIL, PASSWORD)
        mail.select('inbox')

        status, data = mail.search(None, 'ALL')
        for num in data[0].split():
            mail.store(num, '+FLAGS', '\\Seen')

        print(f"{len(data[0].split())} correos marcados como vistos.")

if __name__ == "__main__":
    marcar_todos_como_vistos()