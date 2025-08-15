#!/usr/bin/env python3
"""
Create a KMZ from photos. Each placemark is numbered (1, 2, 3, ...)
and shows an HTML description with an embedded KMZ image path plus extracted EXIF details.

This version:
- Plots EVERY image (even if GPS is missing).
- Uses tiny "jitter" so identical/missing coords never merge into one pin.
- Uses <img style="max-width:500px;" src="files/<image>"> inside the balloon.

Usage:
  python photos_to_kmz.py /path/to/photos output.kmz --name "My Photo Pins"

Requires: Pillow (PIL)
  pip install Pillow
"""

import argparse
import datetime as dt
import math
import re
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image, ExifTags

# ---- EXIF helpers -----------------------------------------------------------

# Build a map from EXIF tag id to human name
EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
GPS_TAGS  = {v: k for k, v in ExifTags.GPSTAGS.items()}

def _rational_to_float(x):
    """Pillow may return tuple(num, den) or Fraction-like objects."""
    try:
        if hasattr(x, "numerator") and hasattr(x, "denominator"):
            return x.numerator / x.denominator
        if isinstance(x, tuple) and len(x) == 2:
            num, den = x
            return float(num) / float(den) if den else float("nan")
        return float(x)
    except Exception:
        return float("nan")

def _dms_to_deg(dms, ref):
    """Convert EXIF DMS tuple to signed decimal degrees."""
    if not dms or len(dms) != 3:
        return None
    d, m, s = (_rational_to_float(part) for part in dms)
    if any(v != v for v in (d, m, s)):  # NaN check
        return None
    deg = d + (m / 60.0) + (s / 3600.0)
    if ref in ["S", "W"]:
        deg = -deg
    return deg

def _format_exposure(val):
    """Pretty-print exposure time."""
    v = _rational_to_float(val)
    if v != v or v <= 0:
        return None
    if v >= 1:
        return f"{v:.1f}s"
    denom = round(1 / v)
    return f"1/{denom}s"

def _format_focal_length(val):
    v = _rational_to_float(val)
    if v != v or v <= 0:
        return None
    return f"{v:.0f} mm"

def _format_altitude(val):
    v = _rational_to_float(val)
    if v != v:
        return None
    return f"{v:.1f} m"

def extract_exif(img_path: Path):
    """
    Return dict with common fields extracted from EXIF (if present):
    lat, lon, when, altitude, make, model, lens, focal_length, exposure, iso
    """
    try:
        with Image.open(img_path) as im:
            exif = im._getexif() or {}
    except Exception:
        return {}

    # Remap keys to tag names
    exif_named = {}
    for tag_id, value in exif.items():
        name = ExifTags.TAGS.get(tag_id, tag_id)
        exif_named[name] = value

    gps_info = exif_named.get("GPSInfo", {})
    if gps_info:
        gps_named = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
    else:
        gps_named = {}

    # Lat/Lon
    lat = lon = None
    if "GPSLatitude" in gps_named and "GPSLatitudeRef" in gps_named:
        lat = _dms_to_deg(gps_named["GPSLatitude"], gps_named["GPSLatitudeRef"])
    if "GPSLongitude" in gps_named and "GPSLongitudeRef" in gps_named:
        lon = _dms_to_deg(gps_named["GPSLongitude"], gps_named["GPSLongitudeRef"])

    # Altitude (optional)
    altitude = None
    if "GPSAltitude" in gps_named:
        altitude = _rational_to_float(gps_named["GPSAltitude"])

    # Timestamp (DateTimeOriginal preferred)
    dt_str = exif_named.get("DateTimeOriginal") or exif_named.get("DateTime")
    when = None
    if isinstance(dt_str, str):
        try:
            when = dt.datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass

    # Other useful EXIF
    make  = exif_named.get("Make")
    model = exif_named.get("Model")
    lens  = exif_named.get("LensModel") or exif_named.get("LensMake")
    focal_length = exif_named.get("FocalLength")
    exposure     = exif_named.get("ExposureTime")
    iso          = exif_named.get("ISOSpeedRatings") or exif_named.get("PhotographicSensitivity")

    # Normalize/format
    focal_length_str = _format_focal_length(focal_length) if focal_length is not None else None
    exposure_str     = _format_exposure(exposure) if exposure is not None else None
    altitude_str     = _format_altitude(altitude) if altitude is not None else None

    return {
        "lat": lat,
        "lon": lon,
        "when": when,
        "altitude": altitude_str,
        "make": make,
        "model": model,
        "lens": lens,
        "focal_length": focal_length_str,
        "exposure": exposure_str,
        "iso": str(iso) if iso is not None else None,
    }

# ---- KML helpers ------------------------------------------------------------

KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS  = "http://www.google.com/kml/ext/2.2"

ET.register_namespace("", KML_NS)
ET.register_namespace("gx", GX_NS)

def kml_elem(tag, **attrs):
    return ET.Element(f"{{{KML_NS}}}{tag}", attrs)

def kml_sub(parent, tag, text=None, **attrs):
    e = ET.SubElement(parent, f"{{{KML_NS}}}{tag}", attrs)
    if text is not None:
        e.text = text
    return e

def make_style(style_id="photo-pin"):
    """Simple yellow pushpin with slightly larger scale."""
    style = kml_elem("Style", id=style_id)
    icon_style = kml_sub(style, "IconStyle")
    kml_sub(icon_style, "scale", "1.2")
    icon = kml_sub(icon_style, "Icon")
    kml_sub(icon, "href", "http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png")
    label_style = kml_sub(style, "LabelStyle")
    kml_sub(label_style, "scale", "0.8")
    return style

def _fmt_coord(v):
    return f"{v:.6f}" if isinstance(v, (int, float)) else ""

def build_balloon_html(kmz_rel_path: str, name: str, meta: dict, has_gps: bool):
    """
    Build HTML description with the exact <img style="max-width:500px;" src="files/<image>">,
    then a small details section extracted from EXIF.
    """
    when = meta.get("when")
    when_str = when.strftime("%Y-%m-%d %H:%M:%S") if when else ""
    lat = meta.get("lat")
    lon = meta.get("lon")
    lat_s = _fmt_coord(lat) if lat is not None else ""
    lon_s = _fmt_coord(lon) if lon is not None else ""

    # Optional extras
    rows = []
    if when_str: rows.append(("Taken", when_str))
    if has_gps and lat_s and lon_s:
        rows.append(("Coordinates", f"{lat_s}, {lon_s}"))
    else:
        rows.append(("GPS", "Not available"))
    if meta.get("altitude"): rows.append(("Altitude", meta["altitude"]))
    cam_bits = " ".join([b for b in [meta.get("make"), meta.get("model")] if b])
    if cam_bits: rows.append(("Camera", cam_bits))
    if meta.get("lens"): rows.append(("Lens", meta["lens"]))
    if meta.get("focal_length"): rows.append(("Focal Length", meta["focal_length"]))
    if meta.get("exposure"): rows.append(("Exposure", meta["exposure"]))
    if meta.get("iso"): rows.append(("ISO", meta["iso"]))

    details_html = ""
    if rows:
        parts = ['<dl style="margin:8px 0 0 0;">']
        for k, v in rows:
            parts.append(f'<dt style="font-weight:bold;float:left;width:120px;clear:left;">{k}</dt>')
            parts.append(f'<dd style="margin:0 0 4px 130px;">{v}</dd>')
        parts.append("</dl>")
        details_html = "".join(parts)

    body = f"""
        <![CDATA[
        <div style="font-family:Arial,Helvetica,sans-serif; max-width: 620px;">
          <h3 style="margin:0 0 8px 0;">{name}</h3>
          <img style="max-width:500px;" src="{kmz_rel_path}">
          {details_html}
        </div>
        ]]>
    """
    return textwrap.dedent(body).strip()

def add_placemark(folder, name, lat, lon, html_description, when=None, style_url="#photo-pin"):
    pm = kml_sub(folder, "Placemark")
    kml_sub(pm, "name", name)
    if style_url:
        kml_sub(pm, "styleUrl", style_url)
    kml_sub(pm, "description", html_description)
    if when:
        time_elem = kml_sub(pm, "TimeStamp")
        kml_sub(time_elem, "when", when.strftime("%Y-%m-%dT%H:%M:%S"))
    point = kml_sub(pm, "Point")
    kml_sub(point, "coordinates", f"{lon:.8f},{lat:.8f},0")
    return pm

# ---- Jitter utilities so pins never merge -----------------------------------

def meters_to_deg_lat(meters: float) -> float:
    # ~111,320 meters per degree latitude
    return meters / 111_320.0

def meters_to_deg_lon(meters: float, lat_deg: float) -> float:
    # meters per degree longitude shrinks by cos(latitude)
    lat_rad = math.radians(lat_deg)
    meters_per_deg = 111_320.0 * math.cos(lat_rad)
    meters_per_deg = meters_per_deg if meters_per_deg > 1e-6 else 1e-6
    return meters / meters_per_deg

def offset_duplicate(lat: float, lon: float, dup_index: int) -> tuple[float, float]:
    """
    For the Nth duplicate at the same coordinate (dup_index >= 1),
    return a slightly offset (lat, lon) so pins don't overlap.
    Golden-angle spiral with ~1.5 m * sqrt(n) spacing.
    """
    if dup_index <= 0:
        return lat, lon
    base_spacing_m = 1.5
    radius_m = base_spacing_m * math.sqrt(dup_index)
    angle = (dup_index * 137.508) % 360.0
    ang_rad = math.radians(angle)
    d_north = radius_m * math.cos(ang_rad)
    d_east  = radius_m * math.sin(ang_rad)
    dlat = meters_to_deg_lat(d_north)
    dlon = meters_to_deg_lon(d_east, lat)
    return lat + dlat, lon + dlon

# ---- Main pipeline ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a KMZ of geotagged photos.")
    parser.add_argument("photos_dir", type=Path, help="Directory containing images (jpg/jpeg/png/tif).")
    parser.add_argument("output_kmz", type=Path, help="Output KMZ file path.")
    parser.add_argument("--name", default="Photo Placemarks", help="Folder/document name in Google Earth.")
    parser.add_argument("--glob", default="*.jpg,*.jpeg,*.png,*.tif,*.tiff", help="Comma-separated file globs.")
    args = parser.parse_args()

    globs = [g.strip() for g in args.glob.split(",") if g.strip()]
    img_paths = []
    for g in globs:
        img_paths.extend(sorted(args.photos_dir.rglob(g)))

    if not img_paths:
        print("No images found. Check --glob and directory.", file=sys.stderr)
        sys.exit(1)

    # KML root
    kml = kml_elem("kml")
    doc = kml_sub(kml, "Document")
    kml_sub(doc, "name", args.name)
    doc.append(make_style("photo-pin"))

    folder = kml_sub(doc, "Folder")
    kml_sub(folder, "name", args.name)

    # Embedded images will be stored inside KMZ under "files/"
    image_folder_in_kmz = "files/"
    added = 0
    items_for_zip = []

    pin_idx = 1  # sequential numbering for pins

    # Track occurrences of identical coordinates to offset duplicates.
    # Group "identical" at 7 decimals (~1 cm).
    coord_counts: dict[tuple[float, float], int] = {}

    # Anchor for non-GPS images: first geotagged coordinate if available
    anchor_latlon = None
    non_gps_counter = 0  # to jitter non-GPS pins around anchor (or 0,0)

    # First pass: find first geotagged coordinate to use as anchor for non-GPS images
    for p in img_paths:
        meta = extract_exif(p)
        lat, lon = meta.get("lat"), meta.get("lon")
        if lat is not None and lon is not None:
            anchor_latlon = (lat, lon)
            break

    for p in img_paths:
        meta = extract_exif(p)
        orig_lat, orig_lon = meta.get("lat"), meta.get("lon")
        when = meta.get("when")

        has_gps = (orig_lat is not None and orig_lon is not None)

        # Determine plotting coordinate (with jitter handling)
        if has_gps:
            key = (round(orig_lat, 7), round(orig_lon, 7))
            dup_index = coord_counts.get(key, 0)  # 0 for first, 1 for second, ...
            coord_counts[key] = dup_index + 1
            lat, lon = offset_duplicate(orig_lat, orig_lon, dup_index)
        else:
            # No GPS: place near anchor (first geotag found), else around (0,0)
            base_lat, base_lon = anchor_latlon if anchor_latlon is not None else (0.0, 0.0)
            lat, lon = offset_duplicate(base_lat, base_lon, non_gps_counter)
            non_gps_counter += 1

        # Build embedded image path in KMZ and use it in the <img> tag
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", p.name)
        rel_path_in_kmz = image_folder_in_kmz + safe_name
        kmz_img_src = rel_path_in_kmz  # e.g., "files/IMG_0097.JPG"

        # Build HTML description with embedded image path + details
        html_desc = build_balloon_html(kmz_rel_path=kmz_img_src, name=str(pin_idx), meta=meta, has_gps=has_gps)

        # Add placemark
        add_placemark(
            folder,
            name=str(pin_idx),
            lat=lat,
            lon=lon,
            html_description=html_desc,
            when=when,
            style_url="#photo-pin",
        )

        # Always embed the original image into the KMZ so it travels with the file
        items_for_zip.append((p, rel_path_in_kmz))
        added += 1
        pin_idx += 1

    # Serialize KML
    kml_bytes = ET.tostring(kml, encoding="utf-8", xml_declaration=True)

    # Write KMZ (zip) with doc.kml + images
    with ZipFile(args.output_kmz, "w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_bytes)
        for src, rel in items_for_zip:
            zf.write(src, rel)

    print(f"Done: {args.output_kmz}  (pins: {added}, non-GPS placed: {non_gps_counter})")

if __name__ == "__main__":
    main()
