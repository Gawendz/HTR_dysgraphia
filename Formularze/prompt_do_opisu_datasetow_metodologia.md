# Informacje dla ChatGPT/Codexa do opisu zbiorów danych i preprocessingu w pracy magisterskiej

Napisz akademicki opis metodologiczny do pracy magisterskiej dotyczącej porównania modeli HTR/OCR/MLLM w zadaniu rozpoznawania ręcznie pisanego tekstu, ze szczególnym uwzględnieniem pisma osób z dysgrafią. Tekst ma dotyczyć własnego zbioru danych oraz adaptacji publicznego zbioru malezyjskiego. Pisz po polsku, formalnie, ale jasno. Nie opisuj zadania jako klasyfikacji dysgrafii, tylko jako rozpoznawanie treści pisma odręcznego. Dysgrafia jest cechą/etykietą grupy badawczej, a nie głównym celem predykcji modelu.

## Własny zbiór danych

Własny zbiór danych został przygotowany na potrzeby pracy. Dane zbierano za pomocą papierowych formularzy z próbkami pisma odręcznego. Formularze były anonimowe i zawierały metadane deklarowane przez uczestnika: płeć, rok urodzenia oraz informację o trudnościach w pisaniu. Uczestnik mógł zaznaczyć brak trudności, dysgrafię lub inne trudności, np. dysleksję albo dysortografię. W opisie należy zaznaczyć, że etykiety grupy trudności wynikają z deklaracji/zaznaczeń na formularzu, a nie z niezależnej diagnozy przeprowadzonej w ramach pracy.

Dane zbierano w szkołach w Augustowie oraz w jednej szkole w Gdańsku. Formularze były przygotowane w trzech wariantach: Zestaw A, Zestaw B i Zestaw C. Każdy wariant zawierał po 8 zdań lub krótkich tekstów do przepisania w osobnych liniach. Teksty obejmowały polskie znaki diakrytyczne, zdania o różnej długości, wielkie litery, cyfry, znaki interpunkcyjne oraz elementy przypominające dane techniczne, takie jak identyfikatory, nazwy plików, adres e-mail, godziny, wartości liczbowe i komunikaty systemowe. Celem takiego doboru treści było uzyskanie próbek przydatnych do oceny modeli rozpoznawania tekstu na poziomie linii.

Formularze zawierały czarne kwadratowe znaczniki w rogach strony. Nie były one elementem dekoracyjnym, tylko służyły do geometrycznego wyznaczania położenia arkusza na zdjęciu/skanie. Dzięki nim możliwe było prostowanie perspektywy oraz mapowanie stałych obszarów formularza, np. pól metadanych i linii z pismem odręcznym. W opisie można podkreślić, że formularz został zaprojektowany tak, aby ułatwić późniejszą automatyzację przetwarzania.

W końcowej wersji własny zbiór obejmuje 320 formularzy. Na poziomie uczestników przyjęto następujące grupy: 212 osób bez zadeklarowanych trudności w pisaniu, 71 osób z dysgrafią oraz 37 osób z innymi trudnościami. W kategorii "inne trudności" dysortografia stanowi 2 z 37 przypadków. Dla płci, po pominięciu rekordów nieokreślonych w wykresach płci, uzyskano 202 mężczyzn i 116 kobiet. Roczniki uczestników obejmują głównie lata 2005-2014, z największą reprezentacją roczników 2009 i 2010.

Każdy formularz mógł dać maksymalnie 8 obrazów linii, więc łącznie istniało 2560 potencjalnych pól linii. Po walidacji i odrzuceniu pustych albo nieprzydatnych próbek do eksportu trafiło 2313 obrazów linii. W podziale na grupy trudności liczba wyeksportowanych obrazów linii wynosi: 1594 dla grupy bez trudności, 458 dla grupy dysgrafii oraz 261 dla grupy innych trudności. Dane podzielono na zbiory: treningowy 1844 obrazów, walidacyjny 228 obrazów oraz testowy 241 obrazów. Podział był wykonywany na poziomie formularza/uczestnika, aby linie tej samej osoby nie trafiały jednocześnie do różnych części zbioru.

## Preprocessing własnego zbioru

Preprocessing własnego zbioru miał charakter półautomatyczny. Wejściem były zdjęcia lub skany formularzy zapisane jako pliki PDF/obrazy. Najpierw wykrywano znaczniki narożne formularza i wykonywano korekcję geometryczną, aby sprowadzić stronę do wspólnego układu odniesienia. Następnie, korzystając ze znanego szablonu formularza, wycinano obszary metadanych oraz obszary odpowiadające liniom tekstu.

Pola metadanych, takie jak płeć, rok urodzenia i deklarowane trudności, były wstępnie odczytywane automatycznie lub półautomatycznie, między innymi na podstawie położenia checkboxów i obszaru roku urodzenia. Wyniki te były następnie kontrolowane ręcznie w przygotowanym programie walidacyjnym. Program wyświetlał obraz formularza lub wycinki linii oraz odpowiadające im wpisy w pliku CSV. Umożliwiał szybką korektę metadanych, ground truth oraz statusu próbki.

Obrazy linii były wycinane z uwzględnieniem tego, że uczestnicy nie zawsze mieścili się idealnie w wyznaczonych polach. Dlatego przyjęto wycinki obejmujące szeroki obszar strony, tak aby nie ucinać końcówek tekstu. Dla szczególnych przypadków, w których uczestnik pisał jedną odpowiedź w dwóch liniach, ręcznie rozszerzano obszary wycinania. Jeśli linia była pusta, nieczytelna, przekreślona w sposób uniemożliwiający jednoznaczne ustalenie tekstu albo budziła poważne wątpliwości, mogła otrzymać status "niepewne" lub "wyklucz" i nie była używana w finalnym eksporcie treningowym.

Ground truth ustalano ręcznie na poziomie linii. Zasadą było przepisywanie tego, co rzeczywiście znajduje się w piśmie odręcznym, a nie automatyczne poprawianie tekstu do zdania referencyjnego z formularza. Jeżeli uczestnik pominął znak interpunkcyjny lub znak diakrytyczny, ground truth odzwierciedlał zapis widoczny na obrazie, o ile był możliwy do jednoznacznego odczytania. W przypadkach niejednoznacznych oznaczano próbkę jako niepewną, aby nie wprowadzać do treningu lub ewaluacji arbitralnych etykiet.

Po walidacji przygotowano eksporty zgodne z potrzebami różnych modeli: obrazy linii i pliki manifestów CSV dla eksperymentów ogólnych, strukturę danych dla TrOCR, pliki tekstowe dla Krakena oraz warianty danych wejściowych dla Qwen/Qwen3-VL. Dzięki temu ten sam zbiór mógł być wykorzystany w scenariuszu zero-shot oraz w eksperymentach z dostrajaniem modeli.

## Zbiór malezyjski

Drugim zbiorem użytym w pracy jest publiczny malezyjski zbiór próbek pisma dzieci w wieku szkolnym, określany roboczo jako Malaysian dataset / Potential Dysgraphia Handwriting Dataset of School-Age Children. Zbiór zawiera próbki podzielone na dwie grupy: PD, czyli potencjalna dysgrafia, oraz LPD, czyli niski potencjał dysgrafii. W lokalnej tabeli metadanych znajduje się 498 wpisów: 228 w grupie PD oraz 270 w grupie LPD. Próbki zawierają teksty w języku malajskim.

W kontekście tej pracy zbiór malezyjski nie służy do diagnozowania dysgrafii, lecz do porównania, jak modele rozpoznające tekst radzą sobie na innym zbiorze i innym języku oraz jak zachowują się względem grup PD i LPD. Należy podkreślić, że oznaczenia PD/LPD pochodzą z oryginalnego zbioru i są traktowane jako metadane eksperymentalne.

## Preprocessing zbioru malezyjskiego

Preprocessing zbioru malezyjskiego różnił się od preprocessingu własnych formularzy, ponieważ nie korzystano tu z zaprojektowanego szablonu z markerami narożnymi. Obrazy zostały zaadaptowane ręcznie do postaci próbek liniowych. Wykonano ręczną segmentację linii tekstu oraz przypisano każdej linii odpowiadający jej ground truth. Do tego celu przygotowano roboczy program walidacyjny podobny do programu używanego dla własnego zbioru: po lewej stronie wyświetlano obraz próbki, a po prawej pola z tekstem ground truth i metadanymi.

Po ręcznej segmentacji lokalna wersja zbioru malezyjskiego zawiera 488 obrazów linii z anotacjami w pliku CSV. Każdy rekord obejmuje nazwę pliku wycinka, tekst ground truth oraz status próbki. Podobnie jak w przypadku własnego zbioru, celem było uzyskanie danych na poziomie linii, ponieważ eksperymenty HTR są prowadzone jako rozpoznawanie tekstu z obrazu pojedynczej linii.

## Jak opisać metodologicznie porównanie

W pracy należy przedstawić oba zbiory jako dwa źródła danych o różnym charakterze. Własny zbiór jest polskojęzyczny, zebrany specjalnie do pracy, kontrolowany formularzem i bogaty w polskie znaki oraz różne typy treści. Zbiór malezyjski jest zbiorem zewnętrznym, publicznym, użytym jako dodatkowy punkt odniesienia i materiał do sprawdzenia uogólniania modeli. W obu przypadkach końcową jednostką danych jest obraz pojedynczej linii wraz z transkrypcją ground truth.

W opisie eksperymentów trzeba zaznaczyć, że modele będą oceniane najpierw w scenariuszu zero-shot/pretrained, czyli bez uczenia na danym zbiorze, a następnie po dostrojeniu. Plan porównania obejmuje sprawdzenie wyników na własnym zbiorze i na zbiorze malezyjskim przed fine tuningiem oraz po fine tuningu, aby ocenić, czy dostrojenie poprawia jakość rozpoznawania i czy wpływ ten różni się między pismem standardowym, pismem dysgraficznym oraz innymi grupami trudności.

Unikaj pisania, że model ma klasyfikować, czy ktoś ma dysgrafię. Poprawne sformułowanie: modele rozpoznają treść pisma odręcznego, a analiza wyników jest prowadzona osobno dla grup uczestników, w tym dla osób z dysgrafią.
