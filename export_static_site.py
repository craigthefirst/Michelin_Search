"""
export_static_site.py
=====================
Exports all necessary data, embeddings, and MDN model weights
for the GitHub Pages client-side static application in docs/
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"
DOCS_IMAGES_DIR = DOCS_DIR / "images"

sys.path.append(str(PROJECT_ROOT))

from core.data_loader import (
    build_restaurant_catalog,
    load_sentiment_df,
    load_menus_df,
    load_reviews_df,
    load_images_df,
    load_restaurant_embeddings,
    load_finegrained_embeddings,
    _format_menu_price,
    _clean_text,
    MENU_EMBEDDINGS_JSONL,
    REVIEW_EMBEDDINGS_JSONL,
    FOOD_EMBEDDINGS_JSONL,
    INTERIOR_EMBEDDINGS_JSONL,
)
from algorithms.retrieval import _format_menu_item_display
from algorithms.mdn_regression import MDNScorer, MDN_CHECKPOINT, get_michelin_centroid


def export_mdn_weights():
    print("--- Exporting Folded MDN Weights ---")
    if not MDN_CHECKPOINT.exists():
        print("Warning: MDN checkpoint not found, skipping weights export.")
        return None

    model = MDNScorer.load_from_checkpoint(str(MDN_CHECKPOINT), map_location="cpu")
    model.eval()

    linear_indices = [0, 4, 8, 12]
    bn_indices = [1, 5, 9, 13]
    folded_layers = []

    for l_idx, bn_idx in zip(linear_indices, bn_indices):
        lin = model.mlp[l_idx]
        bn = model.mlp[bn_idx]

        weight = lin.weight.detach()
        bias = lin.bias.detach() if lin.bias is not None else torch.zeros(lin.out_features)

        running_mean = bn.running_mean.detach()
        running_var = bn.running_var.detach()
        eps = bn.eps
        gamma = bn.weight.detach() if bn.weight is not None else torch.ones(lin.out_features)
        beta = bn.bias.detach() if bn.bias is not None else torch.zeros(lin.out_features)

        scale = gamma / torch.sqrt(running_var + eps)
        w_folded = weight * scale.unsqueeze(1)
        b_folded = (bias - running_mean) * scale + beta

        folded_layers.append({
            "weight": [round(float(v), 5) for row in w_folded.numpy() for v in row],
            "bias": [round(float(v), 5) for v in b_folded.numpy()],
            "in_features": lin.in_features,
            "out_features": lin.out_features
        })

    final_lin = model.final_layer
    final_w = [round(float(v), 5) for row in final_lin.weight.detach().numpy() for v in row]
    final_b = [round(float(v), 5) for v in final_lin.bias.detach().numpy()]

    mdn_data = {
        "k": 3,
        "input_dim": 1540,
        "layers": folded_layers,
        "final_layer": {
            "weight": final_w,
            "bias": final_b,
            "in_features": final_lin.in_features,
            "out_features": final_lin.out_features
        }
    }

    out_path = DOCS_DATA_DIR / "mdn_weights.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mdn_data, f)
    print(f"Saved MDN weights to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return mdn_data


def export_catalog_and_details():
    print("--- Exporting Restaurant Catalog & Details ---")
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    catalog_df = build_restaurant_catalog()
    sentiment_df = load_sentiment_df()
    menus_df = load_menus_df()
    reviews_df = load_reviews_df()
    images_df = load_images_df()

    # Pre-index menus and reviews by rest_id
    menus_by_rest = {}
    if not menus_df.empty and "rest_id" in menus_df.columns:
        for row in menus_df.to_dict(orient="records"):
            rid = _clean_text(row.get("rest_id"))
            if not rid: continue
            menus_by_rest.setdefault(rid, []).append({
                "dish_name": _clean_text(row.get("dish_name")),
                "ingredients": _clean_text(row.get("ingredients")),
                "price": _format_menu_price(row.get("price"))
            })

    reviews_by_rest = {}
    if not reviews_df.empty and "rest_id" in reviews_df.columns:
        for row in reviews_df.to_dict(orient="records"):
            rid = _clean_text(row.get("rest_id"))
            if not rid: continue
            reviews_by_rest.setdefault(rid, []).append({
                "text": _clean_text(row.get("text")),
                "rating": _clean_text(row.get("rating")),
                "source": _clean_text(row.get("source"))
            })

    # Find external image folder
    source_images_candidates = [
        PROJECT_ROOT / "data" / "images",
        Path(r"C:\Users\niels\Documents\GitHub\Calm_Octopuses\data\images")
    ]
    source_images_dir = None
    for cand in source_images_candidates:
        if cand.exists():
            source_images_dir = cand
            break

    catalog_list = []
    copied_images = set()

    for row in catalog_df.to_dict(orient="records"):
        rest_id = _clean_text(row.get("rest_id"))
        if not rest_id: continue

        # Sentiment data
        sentiment = None
        if not sentiment_df.empty and rest_id in sentiment_df.index:
            s_row = sentiment_df.loc[rest_id]
            def get_f(col):
                val = s_row.get(col)
                return float(val) if pd.notna(val) else None
            sentiment = {
                "food_quality": get_f("asp_food_quality"),
                "service": get_f("asp_service"),
                "ambiance": get_f("asp_ambiance"),
                "value": get_f("asp_value"),
                "wait_time": get_f("asp_wait_time"),
                "rating": get_f("rating"),
                "review_count": int(s_row.get("review_count", 0)) if pd.notna(s_row.get("review_count")) else 0
            }

        # Handle representative and gallery images
        image_paths = row.get("image_paths", [])
        web_image_paths = []

        if source_images_dir and source_images_dir.exists():
            for p in image_paths[:5]: # up to 5 images per restaurant
                filename = os.path.basename(p)
                source_file = source_images_dir / filename
                if source_file.exists():
                    dest_file = DOCS_IMAGES_DIR / filename
                    if filename not in copied_images:
                        shutil.copy2(source_file, dest_file)
                        copied_images.add(filename)
                    web_image_paths.append(f"images/{filename}")

        rep_img = web_image_paths[0] if web_image_paths else ""

        entry = {
            "rest_id": rest_id,
            "restaurant_name": _clean_text(row.get("restaurant_name")),
            "homepage": _clean_text(row.get("homepage")),
            "borough": _clean_text(row.get("borough")),
            "michelin_category": _clean_text(row.get("michelin_category")),
            "michelin_stars": int(row.get("michelin_stars", 0)),
            "michelin_award": _clean_text(row.get("michelin_award")),
            "michelin_distinction": _clean_text(row.get("michelin_distinction")),
            "michelin_cuisine": _clean_text(row.get("michelin_cuisine")),
            "michelin_location": _clean_text(row.get("michelin_location")),
            "bio_text": _clean_text(row.get("bio_text")),
            "menu_items": row.get("menu_items", []),
            "review_snippets": row.get("review_snippets", []),
            "search_text": _clean_text(row.get("search_text")),
            "representative_image": rep_img,
            "image_paths": web_image_paths,
            "menu_count": len(menus_by_rest.get(rest_id, [])),
            "review_count": len(reviews_by_rest.get(rest_id, [])),
            "sentiment": sentiment
        }
        catalog_list.append(entry)

    # Save catalog.json
    cat_path = DOCS_DATA_DIR / "catalog.json"
    with open(cat_path, "w", encoding="utf-8") as f:
        json.dump(catalog_list, f)
    print(f"Saved catalog to {cat_path} ({len(catalog_list)} restaurants, {cat_path.stat().st_size / 1024:.1f} KB)")
    print(f"Copied {len(copied_images)} images to docs/images/")

    # Save menus.json
    menus_path = DOCS_DATA_DIR / "menus.json"
    with open(menus_path, "w", encoding="utf-8") as f:
        json.dump(menus_by_rest, f)
    print(f"Saved menus to {menus_path} ({len(menus_by_rest)} restaurants, {menus_path.stat().st_size / 1024:.1f} KB)")

    # Save reviews.json
    reviews_path = DOCS_DATA_DIR / "reviews.json"
    with open(reviews_path, "w", encoding="utf-8") as f:
        json.dump(reviews_by_rest, f)
    print(f"Saved reviews to {reviews_path} ({len(reviews_by_rest)} restaurants, {reviews_path.stat().st_size / 1024:.1f} KB)")


def export_embeddings_and_metadata():
    print("--- Exporting Embeddings, Centroid & Metadata ---")
    embeddings_map = load_restaurant_embeddings()
    cleaned_embeddings = {}
    for rid, vec in embeddings_map.items():
        # Round to 4 decimal places for compact JSON transfer
        cleaned_embeddings[rid] = [round(float(x), 4) for x in vec]

    emb_path = DOCS_DATA_DIR / "restaurant_embeddings.json"
    with open(emb_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_embeddings, f)
    print(f"Saved restaurant embeddings to {emb_path} ({len(cleaned_embeddings)} vectors, {emb_path.stat().st_size / 1024:.1f} KB)")

    # Centroid
    centroid = get_michelin_centroid()
    if centroid is not None:
        cent_path = DOCS_DATA_DIR / "centroid.json"
        with open(cent_path, "w", encoding="utf-8") as f:
            json.dump([round(float(x), 4) for x in centroid.numpy()], f)
        print(f"Saved centroid to {cent_path}")

    # Metadata
    meta_src = PROJECT_ROOT / "data" / "embeddings" / "restaurant_metadata.json"
    if meta_src.exists():
        shutil.copy2(meta_src, DOCS_DATA_DIR / "metadata.json")
        print(f"Copied metadata to {DOCS_DATA_DIR / 'metadata.json'}")

    # Menu item embeddings for fine-grained dish search
    if MENU_EMBEDDINGS_JSONL.exists():
        menu_emb_records = []
        with open(MENU_EMBEDDINGS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    rid = _clean_text(data.get("restaurant_id", data.get("rest_id")))
                    vec = data.get("vector")
                    dish_name = _clean_text(data.get("dish_name"))
                    price = _clean_text(data.get("price"))
                    display_text = _format_menu_item_display(dish_name, price=price) or _clean_text(data.get("text"))
                    search_text = " ".join([dish_name, _clean_text(data.get("text"))]).strip()
                    if rid and vec and display_text:
                        menu_emb_records.append({
                            "rest_id": rid,
                            "display_text": display_text,
                            "search_text": search_text,
                            "vector": [round(float(x), 4) for x in vec]
                        })
                except Exception:
                    continue

        menu_emb_path = DOCS_DATA_DIR / "menu_embeddings.json"
        with open(menu_emb_path, "w", encoding="utf-8") as f:
            json.dump(menu_emb_records, f)
        print(f"Saved menu embeddings to {menu_emb_path} ({len(menu_emb_records)} items, {menu_emb_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    print("=== Starting Calm Octopuses Static Export ===")
    export_catalog_and_details()
    export_embeddings_and_metadata()
    export_mdn_weights()
    print("=== Export Complete! ===")
