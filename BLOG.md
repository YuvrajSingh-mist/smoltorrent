# How Devices Find Each Other Without IPs: The *[mDNS](#quick-reference) + [AWDL](#quick-reference)* Story

by **Yuvraj Singh** · 18 minute read

---

I have four Raspberry Pis and a Mac, all on the same WiFi network and I wanted the Mac to find the Pis automatically, no hardcoded IPs anywhere, no `/etc/hosts`, no `ssh pi@192.168.1.something`. I just wanted them to appear to my Mac.


What follows is the actual conversation that got me there, reconstructed from my chat history. I've kept the questions as I asked them, including the dumb ones.

This covers the basics of a technology called *[mDNS](#quick-reference)* and *[Zeroconf](#quick-reference)* and how [AirDrop](#quick-reference) uses it to find nearby devices. It also covers the Swift [sidecar](#quick-reference) that [*smoltorrent*](https://yuvrajsingh-mist.github.io/smoltorrent/docs.html) uses to talk to Apple's [AWDL](#quick-reference) peer-to-peer WiFi stack, and how the whole thing is logged. 

And in order to understand the concpet pratcially, we will be taking code snippets from a modified implementation of an amazing library called [*grove*](https://github.com/swarnim-j/grove) built by -

---

**Me: ok so I want the Pis to show up on the Mac without me knowing their IPs. how does something like [AirDrop](#quick-reference) do that? like it just finds your phone...**

*Teacher: Good question to start with because [AirDrop](#quick-reference) is actually three or four different technologies stacked on top of each other. But the foundation is something called [Zeroconf](#quick-reference), zero configuration networking. Let's start there.*

*[Zeroconf](#quick-reference) is not one protocol, it's a suite of three things that work together:*

![Zeroconf Suite](diagrams/png/01_zeroconf_suite.png)

*For your use case, the second one is what matters. [mDNS](#quick-reference) is the engine.*

---

**Me: ok but what's [mDNS](#quick-reference)?**

*Teacher: Regular DNS works like a phone book with a librarian. You ask the librarian (a DNS server at say 8.8.8.8), they look it up and tell you the IP.*

![Regular DNS Flow](diagrams/png/02_regular_dns.png)

*[mDNS](#quick-reference) removes the librarian. Instead of asking one server, you shout into the whole room and whoever knows the answer shouts back.*

![mDNS Multicast Discovery](diagrams/png/03_mdns_multicast.png)

*Every device on the network joins a multicast group at address `224.0.0.251` and listens on UDP port `5353`. When someone sends a question to that address, every device on the LAN hears it. The one that knows the answer responds.*

---

**Me: why UDP and not TCP?**

*Teacher: TCP is a private conversation between exactly two endpoints. You do the three-way handshake, establish a connection, send bytes, close it. You physically cannot send one TCP packet to multiple recipients at once. The protocol doesn't support it.*

![TCP Point-to-Point vs UDP Multicast](diagrams/png/04_tcp_vs_udp_multicast.png)

*Also, discovery packets are tiny. Like a few hundred bytes. If you used TCP for that, the handshake overhead would be larger than the actual message.*

*UDP is **fire and forget**, that is, you toss the packet, and be done with it.*

---

**Me: what's special about that 224.0.0.251 address?**

*Teacher: The whole `224.x.x.x` range is reserved for multicast. `224.0.0.251` is specifically IANA-assigned for [mDNS](#quick-reference). No other protocol uses it.*

*The part that matters more than the address is the TTL, which is hardcoded to 1.
TTL is Time To Live: routers decrement it on every hop, and drop the packet when it hits zero. A TTL of 1 means the packet literally cannot survive being forwarded.*

*Your Pi's [mDNS](#quick-reference) advertisement stays on your LAN. It can't leak to the internet or bleed into another network segment even if your router is misconfigured.*

*So the full picture: UDP, multicast to `224.0.0.251`, port `5353`, TTL=1. The kernel joins the multicast group for you. You just open the socket and start talking.*

---

**Me: ok so how does [*smoltorrent*](https://yuvrajsingh-mist.github.io/smoltorrent/docs.html) actually use this?**

*Teacher: Each Pi, when it starts up, runs a `WorkerAdvertiser`. It announces itself with its rank, IP, port, and hostname as a *service*:*

>*Browsers are the listeners that run in the master to discover workers, and advertisers are the broadcasters that run in the workers to announce themselves.*



[`discovery/grove/_mdns.py`](https://github.com/YuvrajSingh-mist/smoltorrent/blob/master/discovery/grove/_mdns.py)
```python

# discovery/grove/_mdns.py

class WorkerAdvertiser:
    def __init__(self, rank: int, port: int, hostname: Optional[str] = None) -> None:
        host = hostname or socket.gethostname()
        ip = get_local_ip()
        self.zc = Zeroconf()
        self.info = ServiceInfo(
            SERVICE_TYPE, #Service type (custom app name, protocol name, link-local)
            f"smoltorrent-rank-{rank}.{SERVICE_TYPE}", #Instance name
            addresses=[socket.inet_aton(ip)],
            port=port,
            properties={
                b"rank": str(rank).encode(),
                b"hostname": host.encode(),
            },
        )
        self.zc.register_service(self.info, allow_name_change=True)
```

*The master side runs a browser that listens for those announcements:*

[`discovery/grove/_mdns.py`](https://github.com/YuvrajSingh-mist/smoltorrent/blob/master/discovery/grove/_mdns.py)
```python
# discovery/grove/_mdns.py

def WorkerBrowser(timeout: float = 10.0) -> list[dict]:
    found: dict[int, dict] = {}
    lock = threading.Lock()

    class Listener(ServiceListener):
        def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            info = zc.get_service_info(type_, name)
            if info and info.addresses:
                ip = socket.inet_ntoa(info.addresses[0])
                props = {
                    k.decode() if isinstance(k, bytes) else k: v.decode()
                    if isinstance(v, bytes) else v
                    for k, v in info.properties.items()
                }
                rank = int(props.get("rank") or -1)
                with lock:
                    found[rank] = {
                        "ip": ip, "port": info.port,
                        "rank": rank, "hostname": props.get("hostname", ""),
                    }

        def remove_service(self, *_): pass
        def update_service(self, zc, type_, name): self.add_service(zc, type_, name)

    zc = Zeroconf()
    browser = ServiceBrowser(zc, SERVICE_TYPE, Listener())
    time.sleep(timeout)
    browser.cancel()
    zc.close()
    return sorted(found.values(), key=lambda x: x["rank"])
```

*After 10 seconds, the master has this:*

```python
[
    {"ip": "192.168.1.42", "port": 5001, "rank": 1, "hostname": "pi4-1"},
    {"ip": "192.168.1.43", "port": 5002, "rank": 2, "hostname": "pi4-2"},
    ...
]
```

*See? No hardcoded IPs. No config files. The Workers announced themselves through the [mDNS](#quick-reference), and the master (the mac mini) listened.*

>*This was the moment it clicked for me: the cluster could move to a different network, the Pis could get new DHCP leases overnight, and nothing would break. The code just works.*

---

**Me: ok that makes sense for when everything is on the same network but how does [AirDrop](#quick-reference) work when there's literally no router, like on a plane or in a field**

*Teacher: That's where [AWDL](#quick-reference) comes in. Apple Wireless Direct Link is  Apple's own peer-to-peer WiFi that requires no access point at all.*

![Normal WiFi vs AWDL](diagrams/png/05_normal_wifi_vs_awdl.png)

---

**Me: but both devices need to be on the same WiFi channel right? how does that work if there's no router coordinating anything**

*Teacher: This is the clever part. Your WiFi chip can only tune to one channel at a time. [AWDL](#quick-reference) works around that by time-slicing. Your phone is mostly on your normal WiFi channel, but every 100 milliseconds or so it briefly switches to an [AWDL](#quick-reference) channel for about 16 milliseconds, listens for any nearby devices, then switches back.*

![AWDL Channel Time-Slicing](diagrams/png/06_awdl_time_slicing.png)

*Your video call or web browsing doesn't notice. 16ms out of every 100ms is invisible to TCP.*

>*This is why you can discover other devices and transfer file while still being conncected to your normal WiFi network. [AWDL](#quick-reference) is a separate channel that your WiFi chip hops to briefly, then hops back.*

---

**Me: so what channels does it use?**

*Teacher: Three specific ones called [social channels](#quick-reference):*

```
Channel 6   ── 2.4 GHz, most common
Channel 44  ── 5 GHz
Channel 149 ── 5 GHz
```

*Devices cycle through these in a fixed sequence. When two devices happen to be on the same [social channel](#quick-reference) at the same time, they exchange tiny packets called [AWDL](#quick-reference) Action Frames with timing info. One device becomes the coordinator and broadcasts the schedule. Everyone else aligns to it.*

*That's why [AirDrop](#quick-reference) sometimes takes a few seconds to show a nearby device. The two devices' [AWDL](#quick-reference) windows need to overlap and sync up first.*

---

**Me: and then [AirDrop](#quick-reference) uses this to discover the other device?**

*Teacher: Just the discovery part. Once two devices find each other through [AWDL](#quick-reference), the actual file transfer happens over a dedicated WiFi Direct connection at full speed. [AWDL](#quick-reference) is the matchmaker, not the pipe.*

*The full [AirDrop](#quick-reference) flow is four separate phases:*

![AirDrop Transfer Phases](diagrams/png/07_airdrop_phases.png)

---

**Me: ok and [*smoltorrent*](https://yuvrajsingh-mist.github.io/smoltorrent/docs.html) uses [AWDL](#quick-reference) too?**

*Teacher: For the discovery part, yes through [*grove*](https://github.com/swarnim-j/grove). The problem is Python can't talk to [AWDL](#quick-reference) directly. [AWDL](#quick-reference) is exposed through Apple's `Network.framework`, which is a Swift and Objective-C API. There's no Python binding for it and the kernel interfaces are private.*

```
AWDL
├── Device discovery
├── Election of a master node
├── Time synchronization
├── Channel synchronization
├── Time slicing / channel hopping
├── IPv6 link-local networking
└── Data transport
```
>*So, [AWDL](#quick-reference) is the whole stack as above! It does three things, just liek we saw at the very start of this blog about [Zeroconf](#quick-reference) - SD or Service Discovery, Addressing ([mDNS](#quick-reference)), and Transport (WiFi-Direct). [AWDL](#quick-reference) does all three but only for peer-to-peer WiFi.*

>*So, a different approach was needed and thus [*grove*](https://github.com/swarnim-j/grove) was used as a [sidecar](#quick-reference) which uses a Swift binary whioch acts as a bridge between python and [AWDL](#quick-reference).*

---

**Me: ok so how do you get Python to use something only Swift can do?**

*Teacher: You write a small Swift binary that does the [AWDL](#quick-reference) part and have Python talk to it through a file on disk.*

---

**Me: a file on disk?**

*Teacher: A Unix Domain Socket. It looks like a file, it lives in `/tmp/`, but it behaves exactly like a network socket. You read from it, write to it, and data flows between the two processes. Zero actual network traffic, it all happens in kernel memory.*

![TCP Socket vs Unix Domain Socket](diagrams/png/08_tcp_vs_uds.png)

*If you `ls -la` that path, you see it has `s` at the front instead of `-` or `d`:*

```
srwxr-xr-x  1 yuvraj  wheel  0 Jun 9 14:30 /tmp/smoltorrent_discover_12345.sock
^
's' = socket file
```

---

**Me: Got it and how does the Swift binary get created in the first place**

*Teacher: Python compiles it on demand. The `.swift` source file ships with the repo. When you first run discovery, Python checks if the compiled binary exists. If not, it runs `swiftc` to compile it:*

[`discovery/grove/swift/compile.py`](https://github.com/YuvrajSingh-mist/smoltorrent/blob/master/discovery/grove/swift/compile.py)
```python
# discovery/grove/swift/compile.py

def ensure_compiled() -> Path:
    bin_path = binary_path()

    if bin_path.exists() and bin_path.stat().st_mtime >= SWIFT_SRC.stat().st_mtime:
        return bin_path  # already up to date

    if not is_available():
        raise RuntimeError("Swift compiler (swiftc) not found.")

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["swiftc", "-O", "-o", str(bin_path), str(SWIFT_SRC)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Swift compilation failed:\n{result.stderr}")
    return bin_path
```

*The `-O` flag optimizes it, `-o` sets the output path. After that you have a native binary that runs directly on the CPU, same as any compiled C program.*

---

**Me: and then Python just launches it as a subprocess?**

*Teacher: Exactly. Python spawns it, the Swift process creates a [UDS](#quick-reference) and waits for Python to connect, then they talk:*

*Swift side, creates the server socket:*

[`discovery/grove/swift/p2p_helper.swift`](https://github.com/YuvrajSingh-mist/smoltorrent/blob/master/discovery/grove/swift/p2p_helper.swift)
```swift
// discovery/grove/swift/p2p_helper.swift

func createUDS(_ path: String) -> Int32 {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0) // This creates a Unix Domain Socket
    guard fd >= 0 else { fputs("[p2p] Failed to create socket\n", stderr); exit(1) }
    unlink(path)

    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    _ = withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
        path.utf8CString.withUnsafeBufferPointer { src in
            memcpy(ptr, src.baseAddress!, min(src.count, MemoryLayout.size(ofValue: ptr.pointee)))
        }
    }

    let bindResult = withUnsafePointer(to: &addr) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
        }
    }
    guard bindResult == 0 else { fputs("[p2p] Bind failed: \(errno)\n", stderr); exit(1) }
    listen(fd, 1)
    return fd
}

// in runDiscover():
let serverFd = createUDS(controlPath)
let fd = accept(serverFd, nil, nil)
guard fd >= 0 else { fputs("[p2p] Accept failed\n", stderr); exit(1) }
```

*Python side, connects as client:*

[`discovery/grove/transport/p2p.py`](https://github.com/YuvrajSingh-mist/smoltorrent/blob/master/discovery/grove/transport/p2p.py)
```python
# discovery/grove/transport/p2p.py (inside swift_discover)

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
#                       ^^^^^^^^
#                       Not AF_INET (TCP), AF_UNIX (file path)

sock.connect(ctrl_path)   # e.g. /tmp/smoltorrent_discover_12345.sock
```

---

**Me: and then Swift sends discovery results through this socket?**

*Teacher: Yes. For the discovery phase it sends plain text lines, one per device found:*

```
Swift writes:  "ready\n"
Swift writes:  "found My-MacBook abc123 4 train.py\n"
Swift writes:  "lost abc123\n"
```

*Python reads byte by byte, looking for the newline that marks the end of each message:*

[`discovery/grove/transport/p2p.py`](https://github.com/YuvrajSingh-mist/smoltorrent/blob/master/discovery/grove/transport/p2p.py)
```python
# discovery/grove/transport/p2p.py

def read_line(sock: socket.socket) -> str:
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            break
        buf += chunk
    return buf.decode().strip()
```
>*Now, here we used SOCK_STREAM which is TCP-like, thus, the message boundaries wont be maintained right? Yes tru, but here we are anyways reading the received message one-byte at a time.*

---

**Me: hold on, why not just call readline() or read until newline in one shot? reading one byte at a time seems obviously wrong**

*Teacher: `readline()` works on file objects, not raw sockets. And `recv(N)` reads **up to** N bytes, and there's no way to tell it "stop when you hit a newline" without buffering and risking consuming the start of the next message. One byte at a time is the honest version of that buffering. For discovery, which gets maybe five messages across ten seconds, the overhead is nothing. Simplicity wins here.*

---

**Me: ok, what's the full picture then, how does the whole thing fit together?**

*Teacher: Here it is end to end:*

![Python–Swift Sidecar Architecture](diagrams/png/09_python_swift_sidecar.png)

*That `includePeerToPeer = true` flag is the key. It tells Apple's networking framework to browse on the `awdl0` interface instead of the normal `en0` WiFi interface. That's the one line that enables [AWDL](#quick-reference) discovery.*

---

**Me: ok you mentioned binary data earlier. if you're sending tensor weights through a socket you can't use newlines as delimiters right?**

*Teacher: Right. Tensor data is raw bytes. Any byte value from 0 to 255 can appear anywhere. The value 10 in decimal is `\n`. If your model weights happen to contain that byte, a newline-delimited protocol would split your message in half.*

*So for binary data you use [length-prefixed](#quick-reference) framing instead. First you send a 4-byte header that says how many bytes are coming, then you send exactly that many bytes.*

![Binary Length-Prefix Framing](diagrams/png/10_binary_framing.png)

---

**Me: why 4 bytes for the header specifically?**

*Teacher: A 4-byte unsigned integer can represent up to about 4 gigabytes. Enough for any model shard you'd be sending.*

---

**Me: and TCP just gives you all these bytes in order? like you don't need to worry about packets?**

*Teacher: TCP guarantees order and delivery. But there's a thing people get wrong. TCP is a byte stream, not a message protocol^. It has no concept of where one `send()` ends and the next begins.*

![TCP Byte Stream: No Message Boundaries](diagrams/png/11_tcp_byte_stream.png)

*The receiver sees `"HelloWorld"`. There's no boundary. TCP handles retransmission, ordering, congestion. But message boundaries are entirely your problem.*

>^Ok let me explain. TCP is indeed a protocol that guarantees delivery of bytes in order. But it does not guarantee that the bytes you send in one `send()` call will be received in one `recv()` call. The data can be split across multiple packets, and the receiver may receive it in chunks of any size. This is why you need to implement your own framing protocol on top of TCP to delineate messages.

---

**Me: so I have to implement that buffering myself, accumulate bytes until I have enough to parse a full frame?**

*Teacher: Exactly. The receiver keeps a running buffer, checks whether it has enough bytes for a header, reads the length, checks whether it has enough bytes for the full payload, extracts it, removes those bytes from the front, and loops. Like this:*

![TCP Frame Buffer Accumulation](diagrams/png/12_tcp_framing_accumulation.png)

*For text control messages (like discovery), accumulate until `\n`:*

[`discovery/grove/transport/p2p.py`](https://github.com/YuvrajSingh-mist/smoltorrent/blob/master/discovery/grove/transport/p2p.py)
```python
# discovery/grove/transport/p2p.py

def read_line(sock: socket.socket) -> str:
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            break
        buf += chunk
    return buf.decode().strip()
```

*For binary payloads (like shard transfers), read a 4-byte length header first, then read exactly that many bytes:*

*Here's the helper that reads the 4-byte length header and returns the integer:*

[`networking/send_receive.py`](https://github.com/YuvrajSingh-mist/smoltorrent/blob/master/networking/send_receive.py)
```python
# networking/send_receive.py

def _recv_length(sock: socket.socket) -> int:
    hdr = bytearray(4)
    n = sock.recv_into(hdr, 4) #The message length is sent as a 4-byte header
    if not n:
        return 0
    if n < 4:                          # TCP may deliver the header in pieces
        received = n
        while received < 4:
            n = sock.recv_into(memoryview(hdr)[received:], 4 - received)
            if not n:
                raise ConnectionError("Socket closed while reading length header")
            received += n
    return struct.unpack(">I", hdr)[0]  # big-endian uint32 → exact byte count
```
*Here's the full receive_message() that uses it to read a [length-prefixed](#quick-reference) payload and unpickle it:*

```python
def receive_message(sock: socket.socket) -> Any:
    msglen = _recv_length(sock)

    buf = bytearray(msglen)
    view = memoryview(buf)
    received = 0
    while received < msglen:           # loop until every byte is in
        n = sock.recv_into(view[received:], min(65536, msglen - received))
        if not n:
            raise ConnectionError("Socket closed mid-message")
        received += n

    return pickle.loads(buf)
```

*The full picture of what you control versus what TCP controls:*

![Network Layers: Your Job vs TCP's Job](diagrams/png/13_network_layers.png)

*Your one job is framing.*

---

**Me: so [*smoltorrent*](https://yuvrajsingh-mist.github.io/smoltorrent/docs.html) runs both [mDNS](#quick-reference) and [AWDL](#quick-reference)? at the same time?**

*Teacher: Yes, in parallel threads. [mDNS](#quick-reference) works on a normal LAN. [AWDL](#quick-reference) works without any router^. Running both means the cluster works in either situation.*

[`discovery/__init__.py`](https://github.com/YuvrajSingh-mist/smoltorrent/blob/master/discovery/__init__.py)
```python
# discovery/__init__.py

def discover_workers(timeout: float = 10.0) -> list[dict]:
    setup_grove_logging()

    mdns_results: list[dict] = []
    airdrop_results: list[dict] = []

    def run_mdns():
        mdns_results.extend(WorkerBrowser(timeout=timeout))

    def run_airdrop():
        if sys.platform != "darwin":
            return
        try:
            airdrop_results.extend(discover_airdrop_workers(timeout=timeout))
        except Exception:
            pass

    t_mdns = threading.Thread(target=run_mdns, daemon=True)
    t_awdl = threading.Thread(target=run_airdrop, daemon=True)
    t_mdns.start()
    t_awdl.start()
    t_mdns.join()
    t_awdl.join()

    # mDNS has real IP/port, so it wins on rank collision
    merged: dict[int, dict] = {}
    for worker in mdns_results:
        merged[worker["rank"]] = worker
    return sorted(merged.values(), key=lambda x: x["rank"])
```

| Method | Transport | Interface | Needs router |
|--------|-----------|-----------|-------------|
| Python [mDNS](#quick-reference) | `224.0.0.251:5353` UDP | `en0` (WiFi) | Yes |
| Swift [AWDL](#quick-reference) | [AWDL](#quick-reference) multicast, ch 6/44/149 | `awdl0` | No |

>^A [mDNS](#quick-reference) needs some sort of a link-local network like LAN to be able to broadcast packets around.
Without it, its not possible to discover other devices. [AWDL](#quick-reference), provided this link-local network which allows mdns (a.k.a [Bonjour](#quick-reference) on macos) to work without a router. It creates a link-local network that enables [mDNS](#quick-reference) to function even in the absence of a traditional network infrastructure.

---

## Quick reference

| Concept | What it is |
|---------|-----------|
| Zeroconf | Suite of protocols: link-local addressing + mDNS + DNS-SD |
| mDNS | Multicast DNS, replaces DNS server with `224.0.0.251:5353` UDP multicast |
| `224.0.0.251` | Link-local multicast address, TTL=1, never leaves your LAN |
| Port `5353` | IANA-assigned for mDNS |
| UDP not TCP | TCP cannot multicast, discovery packets are tiny and stateless |
| AWDL | Apple peer-to-peer WiFi using time-sliced channel hopping, no router needed |
| Social channels | Channels 6, 44, 149 where AWDL chirps happen every ~100ms |
| Bonjour | Apple's brand name for Zeroconf and mDNS |
| AirDrop | AWDL discovery + Bonjour advertising + WiFi Direct transfer + iCloud identity |
| UDS | Unix Domain Socket, a file path that acts like a TCP socket with zero network overhead |
| Sidecar | Swift binary launched by Python, communicating over a UDS |
| `includePeerToPeer = true` | Swift flag that routes Bonjour over `awdl0` instead of `en0` |
| TCP framing | TCP is a byte stream, you need delimiters or length-prefixed headers to create message boundaries |
| Length prefix | First 4 bytes = message length as LE UInt32, rest = payload |
| `[tag]` convention | Every log line starts with `[module]`, makes grep instant |

---

Getting all the way to `discover_workers()` returning four live IPs felt like a lot of machinery for what ultimately just replaces a config file. But each piece has a specific reason to exist: mDNS for normal LAN conditions, AWDL for the no-router case, the Swift sidecar because Python has no path into Apple's peer-to-peer stack, the UDS because cross-process communication doesn't need a network, the length-prefixed framing because TCP won't draw your message boundaries for you. None of it is accidental.

The full code is in `discovery/grove/`.

---
