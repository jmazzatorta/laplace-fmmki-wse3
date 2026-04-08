**Progresso implementazione FMMKI Laplace su architettura Cerebras WSE-3**

**layout.csl** — Configura una mesh 2D di 512×512 PE (massimo occupabile per fmmki 3D). 
Ad ogni PE è associata una cella foglia dell'octree a 6 livelli, mappata tramite Morton encoding 3D->1D->2D. 
Se la terzina di bit di livello di un PE assumono valore pari a 0, al PE è associato il nodo padre del livello corrente.
Sono definiti 12 colori fabric: 6 per la fase upward (colori 3-8) e 6 per la fase downward (colori 9-14), due per livello. 
Per ogni PE e per ogni livello, vengono associate le route hardware in base al ruolo del PE (padre, figlio, nodo di transito). I PE di transito fanno forwarding puro senza coinvolgere la ALU. 
Il routing segue la struttura dell'octree: i figli inviano a WEST/NORTH verso il padre nell'upward, il padre invia a EAST/SOUTH nel downward.

**pe.csl** — Contiene la logica di computazione e di comunicazione di ogni PE. 

- *P2M (Particle-to-Multipole)*: calcola i coefficienti dell'espansione multipolare a partire dai corpi contenuti nella cella. 
Usa la "ricorrenza cartesiana delle armoniche sferiche solide reali", tramite un loop esterno su m e un loop interno su l. 
I coefficienti sono memorizzati in un buffer interleaved (reale,complesso,reaale,complesso,ecc...) di 162 float (81 coefficienti complessi per P=8).
Nota per Professore: nell'implementazione che mi ha proposto P è una variabile e può assumere un valore in \[1,8\]. Ho preferito hardcodare il massimo valore possibile per non appesantire la computazione. 

- *Upward phase*: la fase upward è interamente determinata da tasks.
Ogni PE padre riceve i coefficienti dai 7 figli remoti tramite un sistema basato su token: il padre invia un token (== ID di livello) sul colore downward, il figlio corrispondente risponde con i propri coefficienti sul colore upward. 
La ricezione usa `@fmovs` asincrono con ping-pong su due buffer alternati. 
Al completamento di ogni ricezione, viene attivato un local task (`m2m_pipeline_task`) che lancia il task che riceve i coefficienti dal figlio successivo sull'altro buffer e contemporaneamente esegue la computazione M2M sul buffer appena riempito. Questo garantisce overlap tra comunicazione e computazione.
Il figlio con indice 0 è locale (coincide con il PE padre stesso) e viene processato separatamente tramite un altra funzione M2M locale.

- *Gestione dei livelli*: dato che i DSD fabric richiedono colore e queue noti a compile-time, mentre il livello corrente è una variabile runtime, la selezione del livello avviene tramite blocchi if-else che chiamano funzioni "hardcodate con livello" come argomento.
Così il compilatore genera il codice per tutti i branch e gli argomenti risultano comptime.
A runtime viene eseguito solo quello corrispondente al livello attivo.
Anche la transizione tra livelli è gestita tramite tasks: quando la pipeline di un livello termina, il task esegue il livello successivo. 

- *Send to parent*: i PE che non sono radice (0,0), dopo aver completato tutti i livelli in cui sono padri, inviano i propri coefficienti al padre tramite sistema di token sul colore upward.

Restano da implementare: M2M translate, downward phase L2L, M2L, P2P.

**host.py** — Genera N corpi con coordinate in [0,1)³, li assegna alle celle tramite coordinate intere, calcola il Morton ID 3D, ordina per Morton, raggruppa per cella, e trasferisce i dati al device via memcpy.
Per ogni PE il buffer contiene: morton_id (u32), count (u32), e i corpi (4 float ciascuno: x, y, z, q) come u32.
