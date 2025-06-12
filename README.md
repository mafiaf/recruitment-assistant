# Recruitment-assistent

Dit FastAPI-project biedt functies voor werving, zoals chat en cv-beheer.
De interface kan rechtsboven worden omgeschakeld naar Engels of Nederlands.

## Installatie

Maak een `.env.development` bestand aan in de hoofdmap met de volgende sleutels:

```bash
MONGO_URI=mongodb://localhost:27017
OPENAI_API_KEY=<your-openai-key>
SESSION_SECRET=<long-random-string>
```

Wanneer de applicatie start probeert deze verbinding te maken met MongoDB. Als de verbinding lukt zie je `mongo_utils ready` in de log. Als de verbinding mislukt schakelt de server over naar **NO-DB**-modus zodat je andere functies kunt testen.

## Projectgeschiedenis

De pagina `/projects` toont eerdere projectbeschrijvingen met hun ranglijsten om wervingsbeslissingen te volgen.

## Tests uitvoeren

1. Installeer Python-afhankelijkheden (vereist netwerktoegang):

```bash
pip install -r requirements.txt
pip install pytest
```

2. Voer de testreeks uit met:

```bash
pytest
```

De tests gebruiken FastAPI's `TestClient` en controleren het login-proces.

## Sessiecookies

Sessiecookies worden ondertekend met `itsdangerous`. Wanneer `ENV` op `production` staat worden cookies gemarkeerd als `Secure` zodat browsers ze alleen via HTTPS versturen.
