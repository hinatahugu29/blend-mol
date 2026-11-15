# Blend Mol マルチOSリリースビルド手順書（AI・開発者向け）

このドキュメントは、**Blend Mol** の Windows / Linux / macOS 向けネイティブライブラリを GitHub Actions でビルドし、成果物をダウンロードして配布用 ZIP（All-in-One 版および各 OS 別版）を構築・整理するまでの標準手順です。
他の AI エージェントや開発者がそのままコマンドを実行して再現できるよう、具体的なパスとコマンドを記載しています。

---

## 全体フロー

1. バージョン更新と GitHub への `push`
2. GitHub Actions CI（[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)）での 3 OS ビルド・自動検証
3. `gh` CLI によるビルド成果物（Artifact）のダウンロード
4. `blend_mol/native/` へのライブラリ配置と `dist/` フォルダ整理
5. [`benchmarks/build_release_zip.py`](../benchmarks/build_release_zip.py) による All-in-One 版および OS 別 ZIP の構築
6. Blender CLI によるマニフェスト検証とユニットテスト実行

---

## 前提条件

* `git` および `gh`（GitHub CLI、ログイン済みであること：`gh auth status` で確認）
* Python（Windows 環境では `py` コマンドを使用）
* ローカル検証用 Blender 5.2 以上（例：`D:\SteamLibrary\steamapps\common\Blender\blender.exe`）

---

## 手順 1: バージョン更新と GitHub への Push

新しいリリースを作る場合は、事前に以下を更新します。

1. [`blend_mol/blender_manifest.toml`](../blend_mol/blender_manifest.toml) の `version` を更新する（例：`0.1.92`）。
   * 基準マニフェストの `platforms` は `["windows-x64", "linux-x64", "macos-arm64", "macos-x64"]` にしておく。
2. [`README.md`](../README.md) のバージョン履歴に 1 行追記する。
3. 変更をコミットして現在ブランチ（または `main`）へ push する。

```powershell
git status --short
git add <変更ファイル>
git commit -m "0.1.xx: <変更概要>"
git push
```

---

## 手順 2: GitHub Actions CI の監視

push 後、GitHub Actions（[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)）が自動起動します。

### CI のジョブ構成

1. **`build-native`**（並列実行）
   * `windows-latest` → `blend_mol_uff.dll` をビルドし、`unittest`（116件）を実行
   * `ubuntu-latest` → `libblend_mol_uff.so` をビルドし、`unittest` を実行
   * `macos-latest` → `aarch64-apple-darwin` と `x86_64-apple-darwin` の両方をビルドし、`lipo -create` で Universal Binary（`libblend_mol_uff.dylib`）を作成して `unittest` を実行
2. **`package`**
   * 3 OS の成果物を `blend_mol/native/` に集約し、`build_release_zip.py --require-all-native` で All-in-One ZIP を生成
3. **`blender-smoke`**
   * `windows-latest` と `ubuntu-latest` の両方で Blender 5.2.2 を起動し、ユニットテスト・スモークテスト 4 種・`validate_distribution.py`・`extension validate` を実行

### 監視コマンド

```powershell
# 最新の Run ID を確認
gh run list --limit 3

# Run 完了まで待機（失敗時は非ゼロで終了）
gh run watch <RUN_ID> --exit-status
```

---

## 手順 3: ビルド成果物のダウンロードと配置

CI が成功したら、作業用の一時ディレクトリに Artifact をダウンロードし、ネイティブライブラリを [`blend_mol/native/`](../blend_mol/native) へコピーします。

```powershell
$runId = "<RUN_ID>"
$tmpDir = "$env:TEMP\blend_mol_ci_$runId"

# Artifact 一式をダウンロード
gh run download $runId -D $tmpDir

# ダウンロードされる主なファイル:
# - $tmpDir\native-windows-x64\blend_mol_uff.dll
# - $tmpDir\native-linux-x64\libblend_mol_uff.so
# - $tmpDir\native-macos-universal\libblend_mol_uff.dylib
# - $tmpDir\blend_mol-release-zip\blend_mol-<version>.zip

# アドオンの native/ フォルダへコピー
New-Item -ItemType Directory -Force -Path "blend_mol\native" | Out-Null
Copy-Item "$tmpDir\native-windows-x64\blend_mol_uff.dll" "blend_mol\native\blend_mol_uff.dll" -Force
Copy-Item "$tmpDir\native-linux-x64\libblend_mol_uff.so" "blend_mol\native\libblend_mol_uff.so" -Force
Copy-Item "$tmpDir\native-macos-universal\libblend_mol_uff.dylib" "blend_mol\native\libblend_mol_uff.dylib" -Force
```

> **補足**: ダウンロード完了後、一時ディレクトリ `$tmpDir` は不要であれば `Remove-Item -Recurse -Force $tmpDir` で削除できます。

---

## 手順 4: `dist/` フォルダ整理と各 OS 向け ZIP の構築

### 4-1. 過去バージョンの ZIP を退避

`dist/` 直下を最新バージョンのみに保つため、古い ZIP を `dist/archive/` へ移動します。

```powershell
$version = "0.1.92"  # blender_manifest.toml の version に合わせる
New-Item -ItemType Directory -Force -Path "dist\archive" | Out-Null
Get-ChildItem "dist" -Filter "blend_mol-*.zip" -File |
  Where-Object { $_.Name -notlike "blend_mol-$version*" } |
  Move-Item -Destination "dist\archive\" -Force
```

### 4-2. All-in-One 版および OS 別 ZIP を生成

[`benchmarks/build_release_zip.py`](../benchmarks/build_release_zip.py) は `--platform` オプションに対応しています。

* `--platform all`（または省略）: 存在するネイティブライブラリ（`--require-all-native` 指定時は 3 OS すべて必須）を同梱した `dist/blend_mol-<version>.zip` を生成
* `--platform windows`: `blend_mol_uff.dll` のみ同梱し、ZIP 内の `blender_manifest.toml` を `platforms = ["windows-x64"]` にして `dist/blend_mol-<version>-windows-x64.zip` を生成
* `--platform linux`: `libblend_mol_uff.so` のみ同梱し、`platforms = ["linux-x64"]` にして `dist/blend_mol-<version>-linux-x64.zip` を生成
* `--platform macos`: `libblend_mol_uff.dylib` のみ同梱し、`platforms = ["macos-arm64", "macos-x64"]` にして `dist/blend_mol-<version>-macos-universal.zip` を生成

```powershell
py benchmarks\build_release_zip.py --require-all-native
py benchmarks\build_release_zip.py --platform windows
py benchmarks\build_release_zip.py --platform linux
py benchmarks\build_release_zip.py --platform macos
```

---

## 手順 5: 最終検証

生成した 4 つの ZIP すべてについて、Blender の Extension バリデーションとテストを実行します。

```powershell
$blender = "D:\SteamLibrary\steamapps\common\Blender\blender.exe"
$version = "0.1.92"

# 1. 全 ZIP のマニフェスト・構造チェック
foreach ($zip in Get-ChildItem "dist\blend_mol-$version*.zip") {
    Write-Host "=== $($zip.Name) ==="
    & $blender --background --factory-startup --command extension validate $zip.FullName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# 2. Windows 向け ZIP 内の DLL ロード・計算一致・フォールバック検証
& $blender --background --factory-startup --python-exit-code 1 --python benchmarks\validate_distribution.py -- "dist\blend_mol-$version-windows-x64.zip"

# 3. 依存なしユニットテスト（116件）
py -m unittest discover -s tests
```

---

## 注意事項（AI エージェント向け）

* **アーカイブ階層**: PowerShell の `Compress-Archive` は個別指定したサブフォルダをフラット化する問題があるため、必ず [`benchmarks/build_release_zip.py`](../benchmarks/build_release_zip.py) を使用して `native/` サブフォルダ構造を維持すること。
* **ランタイム切り替え**: [`blend_mol/native_backend.py`](../blend_mol/native_backend.py) は `sys.platform` で `.dll` / `.so` / `.dylib` を自動判別し、ロード不可時は純 Python バックエンドへ自動フォールバックする。
* **macOS バイナリ**: macOS 用 `.dylib` は CI でビルドされた未署名 Universal Binary であるため、実機環境によっては Gatekeeper 警告や Python フォールバック動作になる可能性がある。
* **リポジトリの `.gitignore`**: `dist/` は `.gitignore` 対象。`blend_mol/native/*.so` や `*.dylib` を Git 管理に含めるかローカルビルド用にするかは、その時点の運用方針に従うこと。
