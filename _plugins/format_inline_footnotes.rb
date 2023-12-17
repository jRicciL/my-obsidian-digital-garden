# frozen_string_literal: true

Jekyll::Hooks.register [:notes], :post_convert do |note|
    getTags(note)
  end
  
  def getTags(note)
    a = note.content.scan(
      /\^\[{1}(.+?\]*)\]/
    )
    
    # note.content.gsub!(

    # ) {|num|}
  end
  