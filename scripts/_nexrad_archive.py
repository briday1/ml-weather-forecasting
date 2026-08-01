"""Small NOAA archive client used only by repository data-generation scripts."""

from __future__ import annotations

import re
import ssl
from dataclasses import dataclass
from datetime import date, datetime, timezone
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import certifi

NEXRAD_BASE_URL = "https://unidata-nexrad-level3.s3.amazonaws.com"
STATION_CATALOG_URL = "https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.txt"
USER_AGENT = "ML-Weather-Forecasting/0.1"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())
_MD5 = re.compile(r"[0-9a-fA-F]{32}")
_ARCHIVE_NAME = re.compile(
    r"^(?P<site>[A-Z0-9]{3})_N0B_"
    r"(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_"
    r"(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})$"
)


@dataclass(frozen=True)
class NexradStation:
    identifier: str
    name: str
    state: str
    country: str
    latitude_deg: float
    longitude_deg: float

    @property
    def archive_id(self) -> str:
        return self.identifier[-3:]


@dataclass(frozen=True)
class ArchiveObject:
    url: str
    filename: str
    size: int
    checksum: str | None
    scan_time: datetime


def request_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, context=TLS_CONTEXT, timeout=60.0) as response:
        return response.read()


def parse_station_catalog(payload: bytes) -> tuple[NexradStation, ...]:
    """Parse NOAA/NCEI's current fixed-width NEXRAD station report."""
    lines = payload.decode("utf-8", errors="strict").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith("NCDCID") and "STNTYPE" in line
        ),
        None,
    )
    if header_index is None:
        raise ValueError("NOAA station report is missing its fixed-width header")
    header = lines[header_index]
    names = (
        "NCDCID",
        "ICAO",
        "WBAN",
        "NAME",
        "COUNTRY",
        "ST",
        "COUNTY",
        "LAT",
        "LON",
        "ELEV",
        "UTC",
        "STNTYPE",
    )
    starts = {name: header.index(name) for name in names}
    ordered = sorted(starts.items(), key=lambda item: item[1])
    stops = {
        name: (ordered[index + 1][1] if index + 1 < len(ordered) else None)
        for index, (name, _) in enumerate(ordered)
    }

    def field(line: str, name: str) -> str:
        return line[starts[name] : stops[name]].strip()

    stations = {}
    for line in lines[header_index + 1 :]:
        if len(line) <= starts["STNTYPE"] or field(line, "STNTYPE") != "NEXRAD":
            continue
        identifier = field(line, "ICAO").upper()
        if len(identifier) != 4:
            continue
        try:
            latitude = float(field(line, "LAT"))
            longitude = float(field(line, "LON"))
        except ValueError:
            continue
        stations[identifier] = NexradStation(
            identifier=identifier,
            name=field(line, "NAME"),
            state=field(line, "ST"),
            country=field(line, "COUNTRY"),
            latitude_deg=latitude,
            longitude_deg=longitude,
        )
    if not stations:
        raise ValueError("NOAA station report did not contain NEXRAD stations")
    return tuple(stations[key] for key in sorted(stations))


def current_stations() -> tuple[NexradStation, ...]:
    return parse_station_catalog(request_bytes(STATION_CATALOG_URL))


def normalize_site(site: str) -> str:
    normalized = site.strip().upper()
    if len(normalized) == 3:
        normalized = f"K{normalized}"
    if len(normalized) != 4 or not normalized.isalnum():
        raise ValueError(f"Invalid four-character radar site: {site}")
    return normalized


def scan_time(filename: str) -> datetime:
    match = _ARCHIVE_NAME.fullmatch(filename)
    if match is None:
        raise ValueError(f"Unexpected NEXRAD archive filename: {filename}")
    return datetime(
        int(match.group("year")),
        int(match.group("month")),
        int(match.group("day")),
        int(match.group("hour")),
        int(match.group("minute")),
        int(match.group("second")),
        tzinfo=timezone.utc,
    )


def _listing_url(prefix: str, continuation_token: str | None = None) -> str:
    query = {"list-type": "2", "prefix": prefix}
    if continuation_token is not None:
        query["continuation-token"] = continuation_token
    return f"{NEXRAD_BASE_URL}/?{urlencode(query)}"


def _parse_listing(
    payload: bytes,
    *,
    prefix: str,
) -> tuple[tuple[ArchiveObject, ...], str | None]:
    root = ElementTree.fromstring(payload)
    objects = []
    for content in root.findall("{*}Contents"):
        key = content.findtext("{*}Key")
        size_text = content.findtext("{*}Size")
        etag = (content.findtext("{*}ETag") or "").strip('"')
        if (
            key is None
            or size_text is None
            or not key.startswith(prefix)
            or key != key.rsplit("/", 1)[-1]
        ):
            raise ValueError("Invalid object returned by the NEXRAD archive")
        checksum = f"md5:{etag.lower()}" if _MD5.fullmatch(etag) else None
        objects.append(
            ArchiveObject(
                url=f"{NEXRAD_BASE_URL}/{quote(key)}",
                filename=key,
                size=int(size_text),
                checksum=checksum,
                scan_time=scan_time(key),
            )
        )
    truncated = root.findtext("{*}IsTruncated") == "true"
    token = root.findtext("{*}NextContinuationToken") if truncated else None
    if truncated and not token:
        raise ValueError("Truncated NEXRAD listing omitted its continuation token")
    return tuple(objects), token


def discover_site_day(
    site: str,
    requested_date: date,
) -> tuple[ArchiveObject, ...]:
    """List every native N0B scan for one site and UTC date."""
    archive_id = normalize_site(site)[-3:]
    prefix = f"{archive_id}_N0B_{requested_date:%Y_%m_%d}_"
    objects = []
    continuation_token = None
    while True:
        page, continuation_token = _parse_listing(
            request_bytes(_listing_url(prefix, continuation_token)),
            prefix=prefix,
        )
        objects.extend(page)
        if continuation_token is None:
            break
    return tuple(sorted(objects, key=lambda item: item.scan_time))


__all__ = [
    "NEXRAD_BASE_URL",
    "STATION_CATALOG_URL",
    "USER_AGENT",
    "ArchiveObject",
    "NexradStation",
    "current_stations",
    "discover_site_day",
    "normalize_site",
    "parse_station_catalog",
    "scan_time",
]
