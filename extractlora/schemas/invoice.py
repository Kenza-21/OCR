"""Target JSON schema for invoice extraction (see cahier des charges, annexe)."""

from pydantic import BaseModel, Field


class Fournisseur(BaseModel):
    nom: str
    adresse: str


class LigneFacture(BaseModel):
    description: str
    quantite: float
    prix_unitaire: float


class Facture(BaseModel):
    type_document: str = Field(default="facture")
    numero_facture: str
    date_emission: str
    fournisseur: Fournisseur
    montant_ht: float
    taux_tva: float
    montant_ttc: float
    lignes: list[LigneFacture]


INVOICE_JSON_SCHEMA = Facture.model_json_schema()
