import os
from pathlib import Path
import extract_msg

# Ruta base donde están las carpetas 01, 02, ..., 10
input_dir = Path("/home/raliagav/Projects/suisca/files/Mayo2025")

# Carpeta de salida para los adjuntos extraídos
output_folder = input_dir / "adjuntos_extraidos"
output_folder.mkdir(exist_ok=True)

print(f"📁 Escaneando directorios dentro de: {input_dir}")

# Recorre carpetas 01 a 10
for i in range(1, 11):
    folder = input_dir / f"{i:02}"
    print(f"🔍 Revisando carpeta: {folder}")

    if not folder.exists():
        print(f"⚠️  Carpeta no encontrada: {folder}")
        continue

    msg_files = list(folder.glob("*.msg"))
    print(f"📦 {len(msg_files)} archivos .msg encontrados en {folder}")

    for file in msg_files:
        print(f"\n📨 Procesando: {file.name}")
        try:
            msg = extract_msg.Message(str(file))
            for att in msg.attachments:
                att_name = att.longFilename or att.shortFilename
                if not att_name:
                    continue

                # Limpiar nombre de archivo
                att_name = os.path.basename(att_name.replace("\\", "/"))
                output_file = output_folder / att_name

                # Si ya existe, renombrar
                if output_file.exists():
                    base, ext = os.path.splitext(att_name)
                    counter = 1
                    while (output_folder / f"{base}_{counter}{ext}").exists():
                        counter += 1
                    output_file = output_folder / f"{base}_{counter}{ext}"

                # ✅ Guardar directamente los bytes
                with open(output_file, "wb") as f:
                    f.write(att.data)
                print(f"✅ Guardado: {output_file}")

        except Exception as e:
            print(f"❌ Error al procesar {file.name}: {e}")

print("\n🎉 Proceso completado.")
