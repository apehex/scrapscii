import scrapy

import scrapscii.unicode

# TARGETS ######################################################################

TARGET_DICT = {
    'animals': ['aardvarks', 'amoeba', 'bats', 'bears', 'beavers', 'birds-land', 'birds-water', 'bisons', 'camels', 'cats', 'cows', 'deer', 'dogs', 'dolphins', 'elephants', 'fish', 'frogs', 'insects/ants', 'insects/bees', 'insects/beetles', 'insects/butterflies', 'insects/caterpillars', 'insects/cockroaches', 'insects/other', 'insects/snails', 'insects/worms', 'horses', 'marsupials', 'monkeys', 'moos', 'other-land', 'other-water', 'rabbits', 'reptiles/alligators', 'reptiles/dinosaurs', 'reptiles/lizards', 'reptiles/snakes', 'rhinoceros', 'rodents/mice', 'rodents/other', 'scorpions', 'spiders', 'wolves', ],
    'art-and-design': ['artists', 'borders', 'celtic', 'dividers', 'egyptian', 'escher', 'famous-paintings', 'fleur-de-lis', 'fractals', 'gender-symbols', 'geometries', 'mazes', 'mona-lisa', 'origamis', 'other', 'patterns', 'pentacles', 'sculptures'],
    'books': ['alice-in-wonderland', 'books', 'dr-seuss', 'harry-potter', 'lord-of-the-rings', 'moomintroll', 'other', 'winnie-the-pooh'],
    'buildings-and-places': ['alcatraz', 'bridges', 'buildings', 'castles', 'church', 'cities', 'fences', 'flags', 'furniture/beds', 'furniture/benches', 'furniture/chairs', 'furniture/office', 'furniture/other', 'furniture/sofas', 'furniture/toilets', 'houses', 'lighthouses', 'maps', 'monuments/arc-de-triomphe', 'monuments/eiffel-tower', 'monuments/mount-rushmore', 'monuments/notre-dame', 'monuments/other', 'monuments/pyramids', 'monuments/sacre-coeur', 'monuments/statue-of-liberty', 'monuments/stonehenge', 'monuments/taj-mahal', 'other', 'temple', 'windmills'],
    'cartoons': ['animaniacs', 'beavis-and-butt-head', 'betty-boop', 'casper', 'felix-the-cat', 'flintstones', 'inspector-gadget', 'jetsons', 'looney-tunes', 'mickey-mouse', 'mighty-mouse', 'mushroom', 'other', 'pink-panther', 'popeye', 'ren-and-stimpy', 'roger-rabbit', 'simpsons', 'smurfs', 'south-park', 'spongebob-squarepants', 'tiny-toon-adventures', 'two-stupid-dogs'],
    'clothing-and-accessories': ['bikinis', 'bra', 'crowns', 'dresses', 'footwear', 'glasses', 'handwears', 'hats', 'nightwears', 'other', 'overalls', 'pants', 'shirts', 'skirts', 'umbrellas', 'underwear'],
    'comics': ['alfred-e-neuman', 'archie', 'asterix', 'batman', 'bloom-county', 'calvin-and-hobbes', 'captain-america', 'dilbert', 'garfield', 'judge-dredd', 'lucky-luke', 'mafalda', 'other', 'peanuts', 'spiderman', 'superman', 'x-men'],
    'computers': ['amiga', 'apple', 'atari', 'bug', 'computers', 'floppies', 'fonts', 'game-consoles', 'joysticks', 'keyboards', 'linux', 'mouse', 'other', 'smileys', 'sun-microsystems'],
    'electronics': ['audio-equipment', 'blender', 'calculators', 'cameras', 'clocks', 'electronics', 'light-bulbs', 'other', 'phones', 'robots', 'stereos', 'televisions'],
    'food-and-drinks': ['apples', 'bananas', 'beers', 'candies', 'chocolates', 'coffee-and-tea', 'drinks', 'ice-creams', 'other'],
    'holiday-and-events': ['4th-of-july', 'birthdays', 'christmas/other', 'christmas/religious', 'christmas/santa-claus', 'christmas/snowmen', 'christmas/trees', 'easter', 'fathers-day', 'fireworks', 'graduation', 'halloween', 'hanukkah', 'luck', 'mothers-day', 'new-year', 'other', 'saint-patricks-day', 'thanksgiving', 'valentine', 'wedding'],
    'logos': ['amnesty-international', 'biohazards', 'caduceus', 'coca-cola', 'hello-kitty', 'jolly-roger', 'kool-aid', 'no-bs', 'no-smoking', 'other', 'peace', 'pillsbury-doughboy', 'playboy', 'recycle', 'television'],
    'miscellaneous': ['abacuses', 'anchors', 'antennas', 'awards', 'badges', 'bones', 'bottles', 'boxes', 'brooms', 'buckets', 'candles', 'chains', 'cigarettes', 'diamonds', 'dice', 'dna', 'feathers', 'fire-extinguishers', 'handcuffs', 'hourglass', 'keys', 'kleenex', 'magnifying-glass', 'mailbox', 'medical', 'money', 'noose', 'one-line', 'other', 'playing-cards', 'signs', 'tools'],
    'movies': ['aladdin', 'bambi', 'beauty-and-the-beast', 'ghostbusters', 'ice-age', 'james-bond', 'lion-king', 'little-mermaid', 'other', 'peter-pan', 'pinocchio', 'pocahontas', 'red-dwarf', 'shrek', 'snow-white', 'spaceballs', 'star-wars', 'tinker-bell', 'toy-story', 'wallace-and-gromit'],
    'music': ['musical-instruments', 'musical-notation', 'musicians/alice-cooper', 'musicians/beatles', 'musicians/elvis-presley', 'musicians/marilyn-manson', 'musicians/metallica', 'musicians/other', 'musicians/pet-shop-boys', 'musicians/pink-floyd', 'musicians/snoop-dogg', 'other', 'pianos'],
    'mythology': ['centaurs', 'devils', 'dragons', 'fairies', 'fantasy', 'ghosts', 'grim-reapers', 'gryphon', 'mermaids', 'monsters', 'mythology', 'phoenix', 'skeletons', 'unicorns'],
    'nature': ['beach', 'camping', 'clouds', 'deserts', 'islands', 'landscapes', 'lightning', 'mountains', 'other', 'rainbow', 'rains', 'snows', 'sun', 'sunset', 'tornado', 'waterfall'],
    'people': ['babies', 'bathing', 'body-parts/brains', 'body-parts/buttocks', 'body-parts/eyes', 'body-parts/footprints', 'body-parts/hand-gestures', 'body-parts/lips', 'couples', 'faces', 'famous/al-pacino', 'famous/albert-einstein', 'famous/charlie-chaplin', 'famous/cher', 'famous/george-washington', 'famous/napoleon', 'famous/other', 'famous/princess-diana', 'famous/sarah-michelle-gellar', 'kiss', 'men', 'native-americans', 'other', 'sexual/men', 'sexual/other', 'sexual/women', 'sleeping', 'tribal-people', 'women'],
    'plants': ['bonsai-trees', 'cactus', 'daffodils', 'dandelions', 'flowers', 'leaf', 'marijuana', 'mushroom', 'other', 'roses'],
    'religion': ['angels', 'buddhism', 'christianity', 'crosses-and-crucifixes', 'hinduism', 'judaism', 'other', 'preachers', 'saints', 'yin-and-yang'],
    'space': ['aliens', 'astronauts', 'moons', 'other', 'planetary-rovers', 'planets', 'satellites', 'spaceships', 'stars', 'telescopes'],
    'sports-and-outdoors': ['baseball', 'basketball', 'billiards', 'bowling', 'boxing', 'bungee-jumping', 'chess', 'cycling', 'dancing', 'darts', 'fencing', 'fishing', 'football', 'golf', 'ice-hockey', 'ice-skating', 'logos', 'nba-logos', 'other', 'rodeo', 'scuba', 'skiing', 'soccer', 'surfing', 'swimming', 'tennis'],
    'television': ['babylon-5', 'barney', 'bear-in-the-big-blue-house', 'dexters-laboratory', 'doctor-who', 'futurama', 'galaxy-quest', 'gumby-and-pokey', 'looney-tunes', 'muppets', 'other', 'pinky-and-the-brain', 'rugrats', 'sesame-street', 'star-trek', 'wallace-and-gromit', 'x-files'],
    'toys': ['balloons', 'beanie-babies', 'dolls', 'other', 'pez', 'teddy-bears'],
    'vehicles': ['airplanes', 'bicycles', 'boats', 'busses', 'cars', 'choppers', 'motorcycles', 'navy', 'other', 'trains', 'trucks'],
    'video-games': ['atomic-bomberman', 'creatures', 'hitman', 'lara-croft', 'max-payne', 'mortal-kombat', 'other', 'pacman', 'pokemon', 'sonic-the-hedgehog', 'zelda'],
    'weapons': ['axes', 'bows-and-arrows', 'explosives', 'guillotines', 'guns', 'knives', 'other', 'shields', 'soldiers', 'swords'],}

# ASCII ARCHIVE ################################################################

class AsciiArtSpider(scrapy.Spider):
    name = 'asciiart'

    # META #####################################################################

    urls = [
        f'https://www.asciiart.eu/{__c}/{__i}'
        for __c, __l in TARGET_DICT.items()
        for __i in __l]

    # SCRAPING #################################################################

    def start_requests(self):
        for __u in self.urls:
            yield scrapy.Request(url=__u, callback=self.parse)

    # PARSING ##################################################################

    def parse(self, response):
        for __item in response.css('div.asciiarts > div'):
            # parse
            __all = __item.css('::text').getall()
            if __all:
                # capture
                __caption = ''.join(__all[:-1])
                __content = __all[-1]
                __labels = response.url.split('/')[3:]
                # format
                yield {
                    'caption': __caption,
                    'content': __content,
                    'labels': ','.join(__t.replace('-', ' ').capitalize() for __t in __labels),
                    'charsets': ','.join(set(scrapscii.unicode.lookup_section(__c) for __c in __content)),
                    'chartypes': ','.join(set(scrapscii.unicode.lookup_category(__c) for __c in __content)),}
