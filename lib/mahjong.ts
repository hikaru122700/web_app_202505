export type Tile = string;

export interface Yaku {
  name: string;
  han: number;
}

export interface AgariOptions {
  isTsumo: boolean;
  bakaze: string;
  jikaze: string;
  isRiichi: boolean;
  isIppatsu: boolean;
  isMenzen: boolean;
}

export interface CalculationResult {
  han: number;
  fu: number;
  score: string;
  yaku: Yaku[];
}

export type WaitingPattern = 'penchan' | 'kanchan' | 'tanki' | 'ryanmen' | 'shanpon';

interface Mentsu {
  type: 'shuntsu' | 'koutsu';
  tiles: Tile[];
}

interface HandDecomposition {
  mentsu: Mentsu[];
  pair: Tile;
}

export const TILES = {
  manzu: ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m'],
  pinzu: ['1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p'],
  souzu: ['1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s'],
  jihai: ['東', '南', '西', '北', '白', '發', '中']
};

export const TILE_DISPLAY: Record<string, string> = {
  '1m': '一萬', '2m': '二萬', '3m': '三萬', '4m': '四萬', '5m': '五萬',
  '6m': '六萬', '7m': '七萬', '8m': '八萬', '9m': '九萬',
  '1p': '①', '2p': '②', '3p': '③', '4p': '④', '5p': '⑤',
  '6p': '⑥', '7p': '⑦', '8p': '⑧', '9p': '⑨',
  '1s': '1索', '2s': '2索', '3s': '3索', '4s': '4索', '5s': '5索',
  '6s': '6索', '7s': '7索', '8s': '8索', '9s': '9索',
  '東': '東', '南': '南', '西': '西', '北': '北', '白': '白', '發': '發', '中': '中'
};

export const TILE_ORDER: Record<string, number> = {
  '1m': 1, '2m': 2, '3m': 3, '4m': 4, '5m': 5, '6m': 6, '7m': 7, '8m': 8, '9m': 9,
  '1p': 11, '2p': 12, '3p': 13, '4p': 14, '5p': 15, '6p': 16, '7p': 17, '8p': 18, '9p': 19,
  '1s': 21, '2s': 22, '3s': 23, '4s': 24, '5s': 25, '6s': 26, '7s': 27, '8s': 28, '9s': 29,
  '東': 31, '南': 32, '西': 33, '北': 34, '白': 35, '發': 36, '中': 37
};

export function sortHand(tiles: Tile[]): Tile[] {
  return [...tiles].sort((a, b) => TILE_ORDER[a] - TILE_ORDER[b]);
}

function parseTile(tile: Tile): [number | null, string | null] {
  if (tile.length === 2) {
    const num = parseInt(tile[0]);
    const suit = tile[1];
    return [num, suit];
  }
  return [null, null];
}

function countTiles(hand: Tile[]): Record<string, number> {
  const tileCounts: Record<string, number> = {};
  hand.forEach(tile => {
    tileCounts[tile] = (tileCounts[tile] || 0) + 1;
  });
  return tileCounts;
}

function isYaochuhai(tile: Tile): boolean {
  if (tile.length === 1) return true;
  const num = parseInt(tile[0]);
  return num === 1 || num === 9;
}

function checkMentsu(tiles: Record<string, number>, count: number): boolean {
  if (count === 0) {
    return Object.values(tiles).every(c => c === 0);
  }

  // 刻子チェック
  for (let tile in tiles) {
    if (tiles[tile] >= 3) {
      const remaining = { ...tiles };
      remaining[tile] -= 3;
      if (checkMentsu(remaining, count - 1)) {
        return true;
      }
    }
  }

  // 順子チェック
  for (let tile in tiles) {
    if (tiles[tile] > 0) {
      const [num, suit] = parseTile(tile);
      if (num && num <= 7) {
        const tile2 = `${num + 1}${suit}`;
        const tile3 = `${num + 2}${suit}`;
        if (tiles[tile2] > 0 && tiles[tile3] > 0) {
          const remaining = { ...tiles };
          remaining[tile]--;
          remaining[tile2]--;
          remaining[tile3]--;
          if (checkMentsu(remaining, count - 1)) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

function checkNormalWinningHand(tileCounts: Record<string, number>): boolean {
  // 雀頭を選択
  for (let tile in tileCounts) {
    if (tileCounts[tile] >= 2) {
      const remaining = { ...tileCounts };
      remaining[tile] -= 2;
      if (checkMentsu(remaining, 4)) {
        return true;
      }
    }
  }
  return false;
}

function getMentsuCombinations(tileCounts: Record<string, number>): HandDecomposition[] {
  const decompositions: HandDecomposition[] = [];

  for (const tile in tileCounts) {
    if (tileCounts[tile] < 2) continue;

    const remainingAfterPair = { ...tileCounts };
    remainingAfterPair[tile] -= 2;

    const mentsuList = buildMentsu(remainingAfterPair);
    mentsuList.forEach(mentsu => {
      if (mentsu.length === 4) {
        decompositions.push({ pair: tile, mentsu });
      }
    });
  }

  return decompositions;
}

function buildMentsu(tileCounts: Record<string, number>): Mentsu[][] {
  const firstTile = Object.keys(tileCounts).find(t => tileCounts[t] > 0);
  if (!firstTile) return [[]];

  const results: Mentsu[][] = [];
  const [num, suit] = parseTile(firstTile);

  // 刻子
  if (tileCounts[firstTile] >= 3) {
    const remaining = { ...tileCounts };
    remaining[firstTile] -= 3;
    buildMentsu(remaining).forEach(rest => {
      results.push([{ type: 'koutsu', tiles: [firstTile, firstTile, firstTile] }, ...rest]);
    });
  }

  // 順子
  if (num && num <= 7 && suit) {
    const t2 = `${num + 1}${suit}`;
    const t3 = `${num + 2}${suit}`;
    if ((tileCounts[t2] || 0) > 0 && (tileCounts[t3] || 0) > 0) {
      const remaining = { ...tileCounts };
      remaining[firstTile]--;
      remaining[t2]--;
      remaining[t3]--;
      buildMentsu(remaining).forEach(rest => {
        results.push([{ type: 'shuntsu', tiles: [firstTile, t2, t3].sort((a, b) => TILE_ORDER[a] - TILE_ORDER[b]) }, ...rest]);
      });
    }
  }

  return results;
}

function isSevenPairs(tileCounts: Record<string, number>): boolean {
  return Object.values(tileCounts).filter(count => count === 2).length === 7;
}

export function isWinningHand(hand: Tile[]): boolean {
  const tileCounts = countTiles(hand);

  // 七対子チェック
  if (isSevenPairs(tileCounts)) return true;

  // 国士無双チェック
  const yaochuhai = ['1m', '9m', '1p', '9p', '1s', '9s', '東', '南', '西', '北', '白', '發', '中'];
  const hasAllYaochuhai = yaochuhai.every(tile => tileCounts[tile] >= 1);
  if (hasAllYaochuhai) return true;

  // 通常の和了形チェック
  return checkNormalWinningHand(tileCounts);
}

function isTanyao(hand: Tile[]): boolean {
  return hand.every(tile => {
    if (tile.length === 2) {
      const num = parseInt(tile[0]);
      return num >= 2 && num <= 8;
    }
    return false;
  });
}

function detectWaitForDecomposition(decomposition: HandDecomposition, winningTile: Tile): WaitingPattern {
  if (decomposition.pair === winningTile) {
    return 'tanki';
  }

  const mentsu = decomposition.mentsu.find(m => m.tiles.includes(winningTile));
  if (!mentsu) return 'ryanmen';

  if (mentsu.type === 'koutsu') {
    return 'shanpon';
  }

  // 順子の場合
  const nums = mentsu.tiles.map(t => parseInt(t[0])).sort((a, b) => a - b);
  const winNum = parseInt(winningTile[0]);

  if (winNum === nums[1]) return 'kanchan';
  if (winNum === nums[0]) {
    return winNum === 1 ? 'penchan' : 'ryanmen';
  }
  if (winNum === nums[2]) {
    return winNum === 9 ? 'penchan' : 'ryanmen';
  }

  return 'ryanmen';
}

export function detectWaitingPattern(hand: Tile[], winningTile: Tile): WaitingPattern {
  const tileCounts = countTiles(hand);

  // 七対子は単騎待ち扱い
  if (isSevenPairs(tileCounts)) return 'tanki';

  const decompositions = getMentsuCombinations(tileCounts);
  if (decompositions.length === 0) return 'ryanmen';

  let selected: WaitingPattern = 'ryanmen';
  let bestFu = -1;

  decompositions.forEach(dec => {
    const pattern = detectWaitForDecomposition(dec, winningTile);
    const fuValue = pattern === 'ryanmen' || pattern === 'shanpon' ? 0 : 2;
    if (fuValue > bestFu) {
      bestFu = fuValue;
      selected = pattern;
    }
  });

  return selected;
}

function isValuePair(tile: Tile, options: AgariOptions): boolean {
  const isDragon = tile === '白' || tile === '發' || tile === '中';
  const bakazeMap: Record<string, string> = { ton: '東', nan: '南', sha: '西', pei: '北' };
  const jikazeMap: Record<string, string> = { ton: '東', nan: '南', sha: '西', pei: '北' };
  const bakazeTile = bakazeMap[options.bakaze];
  const jikazeTile = jikazeMap[options.jikaze];
  return isDragon || tile === bakazeTile || tile === jikazeTile;
}

function isPinfu(hand: Tile[], winningTile: Tile, options: AgariOptions, waitingPattern?: WaitingPattern): boolean {
  if (!options.isMenzen) return false;

  const tileCounts = countTiles(hand);
  if (isSevenPairs(tileCounts)) return false;

  const decompositions = getMentsuCombinations(tileCounts);
  return decompositions.some(dec => {
    // すべて順子かつ役牌雀頭なし
    if (!dec.mentsu.every(m => m.type === 'shuntsu')) return false;
    if (isValuePair(dec.pair, options)) return false;
    const pattern = waitingPattern ?? detectWaitForDecomposition(dec, winningTile);
    return pattern === 'ryanmen';
  });
}

function detectYakuhai(hand: Tile[], bakaze: string, jikaze: string): Yaku[] {
  const yaku: Yaku[] = [];
  const tileCounts = countTiles(hand);

  // 三元牌
  if (tileCounts['白'] >= 3) yaku.push({ name: '白', han: 1 });
  if (tileCounts['發'] >= 3) yaku.push({ name: '發', han: 1 });
  if (tileCounts['中'] >= 3) yaku.push({ name: '中', han: 1 });

  // 場風と自風
  const bakazeMap: Record<string, string> = { ton: '東', nan: '南', sha: '西' };
  const jikazeMap: Record<string, string> = { ton: '東', nan: '南', sha: '西', pei: '北' };
  const bakazeTile = bakazeMap[bakaze];
  const jikazeTile = jikazeMap[jikaze];

  // 場風と自風が同じ場合は2翻
  if (bakazeTile === jikazeTile && tileCounts[bakazeTile] >= 3) {
    yaku.push({ name: `場風・自風 ${bakazeTile}`, han: 2 });
  } else {
    // 別々の場合は個別に判定
    if (tileCounts[bakazeTile] >= 3) {
      yaku.push({ name: `場風 ${bakazeTile}`, han: 1 });
    }
    if (tileCounts[jikazeTile] >= 3) {
      yaku.push({ name: `自風 ${jikazeTile}`, han: 1 });
    }
  }

  return yaku;
}

function isToitoihou(hand: Tile[]): boolean {
  const tileCounts = countTiles(hand);

  let koutsu = 0;
  for (let tile in tileCounts) {
    if (tileCounts[tile] >= 3) koutsu++;
  }
  return koutsu >= 4;
}

function countAnkou(hand: Tile[]): number {
  const tileCounts = countTiles(hand);

  let ankou = 0;
  for (let tile in tileCounts) {
    if (tileCounts[tile] >= 3) ankou++;
  }
  return Math.min(ankou, 3);
}

function isHonroutou(hand: Tile[]): boolean {
  return hand.every(tile => {
    if (tile.length === 1) return true;
    if (tile.length === 2) {
      const num = parseInt(tile[0]);
      return num === 1 || num === 9;
    }
    return false;
  });
}

function isShouSangen(tileCounts: Record<string, number>): boolean {
  let sangenCount = 0;
  let sangenPair = 0;
  ['白', '發', '中'].forEach(tile => {
    if (tileCounts[tile] >= 3) sangenCount++;
    if (tileCounts[tile] === 2) sangenPair++;
  });
  return sangenCount === 2 && sangenPair === 1;
}

function isHonitsu(hand: Tile[]): boolean {
  const suits = new Set<string>();
  let hasJihai = false;

  hand.forEach(tile => {
    if (tile.length === 2) {
      suits.add(tile[1]);
    } else {
      hasJihai = true;
    }
  });

  return suits.size === 1 && hasJihai;
}

function isChinitsu(hand: Tile[]): boolean {
  const suits = new Set<string>();
  hand.forEach(tile => {
    if (tile.length === 2) {
      suits.add(tile[1]);
    }
  });
  return suits.size === 1 && hand.every(tile => tile.length === 2);
}

export function detectYaku(hand: Tile[], winningTile: Tile, options: AgariOptions, waitingPattern?: WaitingPattern): Yaku[] {
  const yaku: Yaku[] = [];
  const tileCounts = countTiles(hand);

  // リーチ
  if (options.isRiichi) {
    yaku.push({ name: 'リーチ', han: 1 });
    if (options.isIppatsu) {
      yaku.push({ name: '一発', han: 1 });
    }
  }

  // ツモ
  if (options.isTsumo && options.isMenzen) {
    yaku.push({ name: '門前清自摸和', han: 1 });
  }

  // 断么九（タンヤオ）
  if (isTanyao(hand)) {
    yaku.push({ name: '断么九', han: 1 });
  }

  // 平和（ピンフ）
  if (isPinfu(hand, winningTile, options, waitingPattern)) {
    yaku.push({ name: '平和', han: 1 });
  }

  // 役牌
  const yakuhai = detectYakuhai(hand, options.bakaze, options.jikaze);
  yaku.push(...yakuhai);

  // 七対子
  if (isSevenPairs(tileCounts)) {
    yaku.push({ name: '七対子', han: 2 });
  }

  // 対々和
  if (isToitoihou(hand)) {
    yaku.push({ name: '対々和', han: 2 });
  }

  // 三暗刻
  const ankou = countAnkou(hand);
  if (ankou === 3) {
    yaku.push({ name: '三暗刻', han: 2 });
  }

  // 混老頭
  if (isHonroutou(hand)) {
    yaku.push({ name: '混老頭', han: 2 });
  }

  // 小三元
  if (isShouSangen(tileCounts)) {
    yaku.push({ name: '小三元', han: 2 });
  }

  // 混一色
  if (isHonitsu(hand)) {
    yaku.push({ name: '混一色', han: 3 });
  }

  // 清一色
  if (isChinitsu(hand)) {
    yaku.push({ name: '清一色', han: 6 });
  }

  return yaku;
}

export function isAnkou(tile: Tile, winningTile: Tile, isTsumo: boolean, tileCounts: Record<string, number>): boolean {
  // ツモの場合は必ず暗刻扱い
  if (isTsumo) return true;

  // ロン和了で和了牌が刻子を作っている場合は明刻扱い
  const total = tileCounts[tile] || 0;
  if (tile === winningTile && total === 3) {
    return false;
  }

  return true;
}

function calculateMentsuFu(mentsu: Mentsu, winningTile: Tile, isTsumo: boolean, tileCounts: Record<string, number>): number {
  if (mentsu.type === 'shuntsu') return 0;

  const tile = mentsu.tiles[0];
  const yaochu = isYaochuhai(tile);
  const ankou = isAnkou(tile, winningTile, isTsumo, tileCounts);

  if (yaochu) {
    return ankou ? 8 : 4;
  }
  return ankou ? 4 : 2;
}

function calculatePairFu(tile: Tile, options: AgariOptions): number {
  let fu = 0;
  if (tile === '白' || tile === '發' || tile === '中') fu += 2;

  const bakazeMap: Record<string, string> = { ton: '東', nan: '南', sha: '西', pei: '北' };
  const jikazeMap: Record<string, string> = { ton: '東', nan: '南', sha: '西', pei: '北' };
  const bakazeTile = bakazeMap[options.bakaze];
  const jikazeTile = jikazeMap[options.jikaze];

  if (tile === bakazeTile) fu += 2;
  if (tile === jikazeTile) fu += 2;

  return fu;
}

function waitFu(waitingPattern: WaitingPattern): number {
  switch (waitingPattern) {
    case 'penchan':
    case 'kanchan':
    case 'tanki':
      return 2;
    default:
      return 0;
  }
}

function calculateFuForDecomposition(
  decomposition: HandDecomposition,
  waitingPattern: WaitingPattern,
  winningTile: Tile,
  options: AgariOptions,
  tileCounts: Record<string, number>
): number {
  let fu = 20;

  if (options.isTsumo) fu += 2;
  if (!options.isTsumo && options.isMenzen) fu += 10;

  // 雀頭符
  fu += calculatePairFu(decomposition.pair, options);

  // 面子符
  decomposition.mentsu.forEach(m => {
    fu += calculateMentsuFu(m, winningTile, options.isTsumo, tileCounts);
  });

  // 待ち符
  fu += waitFu(waitingPattern);

  // ピンフ判定（平和ツモは20符固定、ロンは30符）
  if (isPinfu([...decomposition.mentsu.flatMap(m => m.tiles), decomposition.pair, decomposition.pair], winningTile, options, waitingPattern)) {
    return options.isTsumo ? 20 : 30;
  }

  // 七対子（25符固定）
  if (isSevenPairs(tileCounts)) {
    return 25;
  }

  return Math.ceil(fu / 10) * 10;
}

export function calculateFu(
  hand: Tile[],
  winningTile: Tile,
  waitingPattern: WaitingPattern,
  options: AgariOptions
): number {
  const tileCounts = countTiles(hand);

  // 七対子は25符固定
  if (isSevenPairs(tileCounts)) {
    return 25;
  }

  const decompositions = getMentsuCombinations(tileCounts);
  if (decompositions.length === 0) {
    // フォールバック（和了形チェック済みなので20符を返す）
    return 20;
  }

  let maxFu = 0;
  decompositions.forEach(dec => {
    const pattern = detectWaitForDecomposition(dec, winningTile);
    if (pattern !== waitingPattern) return;
    const fu = calculateFuForDecomposition(dec, pattern, winningTile, options, tileCounts);
    if (fu > maxFu) maxFu = fu;
  });

  // 待ち形に一致する分解がない場合は最大符の分解を使用
  if (maxFu === 0) {
    decompositions.forEach(dec => {
      const pattern = detectWaitForDecomposition(dec, winningTile);
      const fu = calculateFuForDecomposition(dec, pattern, winningTile, options, tileCounts);
      if (fu > maxFu) maxFu = fu;
    });
  }

  return maxFu;
}

export function calculateFinalScore(han: number, fu: number, isOya: boolean, isTsumo: boolean): string {
  let baseScore: number;

  // 満貫以上
  if (han >= 5) {
    if (han <= 5) baseScore = 2000; // 満貫
    else if (han <= 7) baseScore = 3000; // 跳満
    else if (han <= 10) baseScore = 4000; // 倍満
    else if (han <= 12) baseScore = 6000; // 三倍満
    else baseScore = 8000; // 役満
  } else {
    // 通常計算
    baseScore = fu * Math.pow(2, 2 + han);
  }

  let score: string;
  if (isOya) {
    if (isTsumo) {
      const perPerson = Math.ceil(baseScore * 2 / 100) * 100;
      score = `${perPerson}点オール（合計${perPerson * 3}点）`;
    } else {
      score = `${Math.ceil(baseScore * 6 / 100) * 100}点`;
    }
  } else {
    if (isTsumo) {
      const ko = Math.ceil(baseScore / 100) * 100;
      const oya = Math.ceil(baseScore * 2 / 100) * 100;
      score = `子: ${ko}点、親: ${oya}点（合計${ko * 2 + oya}点）`;
    } else {
      score = `${Math.ceil(baseScore * 4 / 100) * 100}点`;
    }
  }

  return score;
}

export function calculateScore(
  hand: Tile[],
  winningTile: Tile,
  options: AgariOptions
): CalculationResult | { error: string } {
  if (hand.length !== 13) {
    return { error: '手牌は13枚必要です' };
  }

  if (!winningTile) {
    return { error: '和了牌を選択してください' };
  }

  // 和了形チェック
  const fullHand = [...hand, winningTile];
  if (!isWinningHand(fullHand)) {
    return { error: '和了形ではありません' };
  }

  // 待ち形判定
  const waitingPattern = detectWaitingPattern(fullHand, winningTile);

  // 役の判定
  const yaku = detectYaku(fullHand, winningTile, options, waitingPattern);

  if (yaku.length === 0) {
    return { error: '役がありません' };
  }

  // 翻数計算
  let totalHan = 0;
  yaku.forEach(y => totalHan += y.han);

  // 符計算
  const fu = calculateFu(fullHand, winningTile, waitingPattern, options);

  // 点数計算
  const score = calculateFinalScore(totalHan, fu, options.jikaze === 'ton', options.isTsumo);

  return { han: totalHan, fu, score, yaku };
}
