import imaplib

IMAP_SERVER = 'IMAP.servidor-correo.net'  # Ej: 'imap.gmail.com'
EMAIL = 'rfq.hue@suisca.com'
PASSWORD = '=3s{dr3jKD_34BaA'

def marcar_rango_como_no_vistos(desde=1, hasta=100):
    with imaplib.IMAP4_SSL(IMAP_SERVER) as mail:
        mail.login(EMAIL, PASSWORD)
        mail.select('inbox')

        for uid in range(desde, hasta + 1):
            result, _ = mail.uid('STORE', str(uid), '-FLAGS', '(\\Seen)')
            if result == 'OK':
                print(f"Correo UID {uid} marcado como NO visto.")
            else:
                print(f"Error con UID {uid}")

if __name__ == "__main__":
    marcar_rango_como_no_vistos()
