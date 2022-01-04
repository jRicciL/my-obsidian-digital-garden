# frozen_string_literal: true

# Turns ==something== in Markdown to <mark>something</mark> in output HTML

Jekyll::Hooks.register [:notes], :post_convert do |note|
  format_tag(note)
end

def format_tag(note)
  note.content.gsub!(
    /(\B#\w*[a-zA-Z]+\w*)/, 
    "<span class='tag'>\\1</span>"
  )
end
