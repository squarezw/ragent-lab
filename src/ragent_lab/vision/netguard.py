"""Checks that a visitor-supplied endpoint URL points at a public https host.

The server makes the request, so without this a custom model URL could reach internal
services (SSRF). The check runs when the job is submitted and again before each request;
redirects are never followed (see compat.new_client).
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

_BLOCKED_SUFFIXES = ('.local', '.localhost', '.internal', '.lan', '.home', '.corp')


class UnsafeURL(ValueError):
    """The URL may not be called; the message is user-facing (Chinese)."""


def host_label(url: str) -> str:
    return urlsplit(url).hostname or url


def _public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip.split('%', 1)[0])
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_global and not addr.is_multicast


def check_public_https(url: str, resolve=socket.getaddrinfo) -> str:
    """Raise UnsafeURL unless url is https on a host that resolves only to public IPs."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        raise UnsafeURL('接口地址格式不正确') from None
    host = (parts.hostname or '').rstrip('.').lower()
    if parts.scheme != 'https':
        raise UnsafeURL('接口地址必须以 https:// 开头')
    if not host or parts.username or parts.password:
        raise UnsafeURL('接口地址格式不正确')
    if host == 'localhost' or host.endswith(_BLOCKED_SUFFIXES) or '.' not in host:
        raise UnsafeURL('接口地址不能指向本机或内网')
    try:
        infos = resolve(host, port or 443, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError):
        raise UnsafeURL(f'无法解析接口域名 {host}') from None
    ips = {info[4][0] for info in infos}
    if not ips or not all(_public(ip) for ip in ips):
        raise UnsafeURL('接口地址不能指向本机或内网')
    return host
