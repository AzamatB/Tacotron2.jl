content = open(metadatapath, "r+") do file
   read(file, String)
end

content = replace(content, r"[àâ]" => 'a')
content = replace(content, r"[èéê]" => 'e')
content = replace(content, 'ü' => 'u')
content = replace(content, '’' => '\'')
content = replace(content, r"[“”]" => '\"')

open(metadatapath, "w") do file
   write(file, content)
end
