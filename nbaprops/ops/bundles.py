"""Model bundles in Postgres, so a fresh runner can load the active model (plan, Stage 11:
"save every trained model ... named by its run_id").

The champion is the model_runs row with status 'champion'; challengers project in shadow
(their projections are stored under their own run_id, but pricing uses the champion's).
"""

import gzip
import tempfile
from pathlib import Path

from sqlalchemy import select, update

from nbaprops.db.models import ModelFile, ModelRun
from nbaprops.models.project import Bundle, register_run

FILES = ("minutes.json", "rates.json", "scenario.json", "calibrator.json", "meta.json")


def save(session, bundle: Bundle, champion: bool = False) -> None:
    """Store a bundle's files and register its run (a challenger unless `champion`)."""
    register_run(session, bundle)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        bundle.save(folder)
        for name in FILES:
            blob = gzip.compress((folder / name).read_bytes())
            existing = session.get(ModelFile, (bundle.run_id, name))
            if existing:
                existing.content = blob
            else:
                session.add(ModelFile(run_id=bundle.run_id, name=name, content=blob))
    session.flush()
    if champion:
        promote(session, bundle.run_id)


def load(session, run_id: str) -> Bundle:
    files = {
        f.name: f.content
        for f in session.scalars(select(ModelFile).where(ModelFile.run_id == run_id))
    }
    missing = [n for n in FILES if n not in files]
    if missing:
        raise FileNotFoundError(f"model {run_id} is missing {missing} in model_files")
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for name, blob in files.items():
            (folder / name).write_bytes(gzip.decompress(blob))
        return Bundle.load(folder)


def promote(session, run_id: str) -> None:
    """Make `run_id` the champion; the previous champion is retired (rollback = promote it
    again: a one-row change, plan Stage 11)."""
    session.execute(
        update(ModelRun)
        .where(ModelRun.status == "champion", ModelRun.run_id != run_id)
        .values(status="retired")
    )
    session.execute(update(ModelRun).where(ModelRun.run_id == run_id).values(status="champion"))


def champion(session) -> str | None:
    return session.scalar(select(ModelRun.run_id).where(ModelRun.status == "champion"))


def stored_challengers(session) -> list[str]:
    """Challengers with stored files: they project in shadow."""
    ids = select(ModelFile.run_id).distinct()
    return list(
        session.scalars(
            select(ModelRun.run_id)
            .where(ModelRun.status == "challenger", ModelRun.run_id.in_(ids))
            .order_by(ModelRun.created_ts)
        )
    )
