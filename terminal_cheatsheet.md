2.1 First-time: create & run the DB container

docker run --name atomix-pg `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=postgres `
  -e POSTGRES_DB=atomix `
  -p 5432:5432 `
  -v atomix_pgdata:/var/lib/postgresql/data `
  -d postgres:16


2.2 Daily: start/stop/restart DB

Start: docker start atomix-pg
Stop: docker stop atomix-pg
Restart: docker restart atomix-pg


2.3 Check DB status & logs

List running containers: docker ps
Follow Postgres logs: docker logs -f atomix-pg
Last 200 lines: docker logs --tail 200 atomix-pg


2.4 Connect to DB (psql inside container)

Interactive: docker exec -it atomix-pg psql -U postgres -d atomix


3) Alembic Migrations
3.1 Initialize Alembic (one-time)

If you havent already:

alembic init alembic

(Then ensure your alembic/env.py is wired to atomix.models.Base and uses DATABASE_URL_SYNC.)
3.2 Create a migration (autogenerate)

alembic revision --autogenerate -m "init"

3.3 Apply migrations

alembic upgrade head

3.4 Inspect migration state

Current revision:

alembic current

History:

alembic history

Rollback one migration:

alembic downgrade -1

If PATH/venv issues occur, run via Python:

python -m alembic upgrade head

4) Run the Backend Server (FastAPI + Uvicorn)
4.1 Run dev server (auto-reload)

uvicorn atomix.main:app --reload

Access:

    Swagger UI: http://127.0.0.1:8000/docs

    OpenAPI JSON: http://127.0.0.1:8000/openapi.json

4.2 Stop server

    Press CTRL+C in the terminal running uvicorn.

4.3 If port 8000 is stuck

Find process using port 8000:

netstat -ano | findstr :8000

Kill it:

taskkill /PID <PID> /F

5) Quick Connection Checks
5.1 Is Postgres listening on 5432?

netstat -ano | findstr :5432

5.2 Quick SQLAlchemy ping (sync)

python -c "from sqlalchemy import create_engine, text; e=create_engine('postgresql+psycopg://postgres:postgres@127.0.0.1:5432/atomix'); print(e.connect().execute(text('select 1')).scalar())"

6) Typical Daily Workflow

    Activate venv:

.venv\Scripts\activate

    Start DB:

docker start atomix-pg

    Apply migrations:

alembic upgrade head

    Run backend:

uvicorn atomix.main:app --reload

    Open Swagger:

    http://127.0.0.1:8000/docs

(Optional) Stop DB when done:

docker stop atomix-pg

7) Docker Compose Alternative (Optional)

If you add a docker-compose.yml:

Start:

docker compose up -d

Stop:

docker compose down

Logs:

docker compose logs -f

8) Common Gotchas

    If migrations hang or connect slowly: prefer 127.0.0.1 over localhost in DB URLs.

    Alembic needs a sync driver (psycopg/psycopg2) even if runtime uses asyncpg.

    If alembic revision --autogenerate doesn’t detect tables, your alembic/env.py probably isn’t importing atomix.models properly.






To update on ec2:
# On EC2
cd ~/Mix-Generator-Backend
git pull origin main

# If requirements changed
source .venv/bin/activate
pip install -r requirements.txt

# If database models changed
alembic upgrade head

# Restart the service
sudo systemctl restart atomix

# On EC2, verify everything:
sudo systemctl status atomix
sudo journalctl -u atomix -n 50 -f  # Watch logs
curl http://localhost:8000/docs  