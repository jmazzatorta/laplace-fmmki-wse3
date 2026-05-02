#!/usr/bin/env cs_python
"""
Diagnostic minimal launcher.
Skippa TUTTI gli H2D e lancia direttamente start_fmm.
Serve per verificare se i PE eseguono almeno il primo simprint.

Atteso:
- Se i PE eseguono start_fmm: vedrai migliaia di "STARTFMM" nel sim.log
- Algoritmicamente fallirà (bodies=0, tabelle math=0), ma il programma deve PARTIRE
- Il D2H finale probabilmente si bloccherà perché unblock_cmd_stream non sarà chiamato
  (start_fmm con bodies=0 stalla in attesa di M2M che non arriveranno)
- L'IMPORTANTE è vedere i print STARTFMM nei log
"""
import argparse
import time
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime,
    MemcpyDataType,
    MemcpyOrder,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="compile output dir")
    parser.add_argument("--cmaddr", help="IP:port for CS system")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout in secondi per il D2H (default 5min)")
    args = parser.parse_args()

    print("=" * 60)
    print("DIAGNOSTIC MINIMAL — skip all H2D, launch start_fmm directly")
    print("=" * 60)

    print("\n[1/3] Loading kernel onto device ...")
    t0 = time.time()
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    runner.load()
    runner.run()
    print(f"      Load+Run wall-clock : {time.time()-t0:.2f} s")

    # Verifica simboli essenziali
    canarino_id = runner.get_id("ptr_canarino")
    forces_id = runner.get_id("ptr_forces_buf")
    print(f"      ptr_canarino     id : {canarino_id}")
    print(f"      ptr_forces_buf   id : {forces_id}")

    print("\n[2/3] Launching start_fmm WITHOUT any H2D (data are all zeros)...")
    print("      Expected: PE printano STARTFMM nei sim.log")
    print("      L'algoritmo stallerà perché i dati sono zeri, ma è OK")
    t0 = time.time()
    try:
        runner.launch("start_fmm", nonblock=False)
        print(f"      start_fmm launch returned in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"      EXCEPTION during launch: {e}")

    print("\n[3/3] Tentativo D2H del canarino (singolo PE)...")
    print(f"      Timeout {args.timeout}s. Se si blocca, premi Ctrl+C.")
    print("      Anche se si blocca, i print STARTFMM dovrebbero essere nel sim.log.")

    t0 = time.time()
    try:
        out = np.zeros(1, dtype=np.uint32)
        runner.memcpy_d2h(
            out, canarino_id,
            0, 0, 1, 1, 1,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            nonblock=False,
            order=MemcpyOrder.ROW_MAJOR,
        )
        print(f"      D2H canarino completed in {time.time()-t0:.2f}s")
        print(f"      canarino value at PE(0,0) = {out[0]}")
    except KeyboardInterrupt:
        print(f"\n      Ctrl+C ricevuto dopo {time.time()-t0:.2f}s")
        print("      D2H non completato (atteso, perché unblock_cmd_stream non arriva)")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 60)
    print("Verifica ora:")
    print("  grep -c STARTFMM sim.log")
    print("Se il count > 0: i PE hanno eseguito start_fmm (SUCCESS).")
    print("Se il count == 0: il programma non parte. Problema strutturale.")

    try:
        runner.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
